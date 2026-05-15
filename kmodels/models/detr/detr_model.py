import keras
from keras import layers, ops, utils

from kmodels.base import BaseModel
from kmodels.base.base_model import hf_num_labels
from kmodels.layers import ImageNormalizationLayer
from kmodels.models.detr.detr_layers import (
    DETRExpandQueryEmbedding,
    DETRFlattenFeatures,
    DETRMultiHeadAttention,
    DETRPositionEmbeddingSine,
)

from .config import DETR_CONFIG, DETR_WEIGHTS
from .convert_detr_torch_to_keras import transfer_detr_weights


def detr_encoder_layer(
    x,
    pos_embed,
    hidden_dim,
    num_heads,
    dim_feedforward,
    dropout_rate=0.1,
    block_prefix="encoder_layers_0",
):
    """Single DETR transformer encoder layer.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
    """
    self_attn = DETRMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        block_prefix=f"{block_prefix}_self_attn",
        name=f"{block_prefix}_self_attn",
    )

    q = k = layers.Add(name=f"{block_prefix}_sa_qk_add")([x, pos_embed])
    attn_output = self_attn(q, k, x)
    attn_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_sa_drop")(
        attn_output
    )
    x = layers.Add(name=f"{block_prefix}_sa_residual")([x, attn_output])
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_self_attn_layer_norm",
    )(x)

    ff_output = layers.Dense(
        dim_feedforward,
        activation="relu",
        name=f"{block_prefix}_fc1",
    )(x)
    ff_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_ff_drop")(ff_output)
    ff_output = layers.Dense(
        hidden_dim,
        name=f"{block_prefix}_fc2",
    )(ff_output)
    x = layers.Add(name=f"{block_prefix}_ff_residual")([x, ff_output])
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_final_layer_norm",
    )(x)

    return x


def detr_decoder_layer(
    x,
    memory,
    pos_embed,
    query_pos,
    hidden_dim,
    num_heads,
    dim_feedforward,
    dropout_rate=0.1,
    block_prefix="decoder_layers_0",
):
    """Single DETR transformer decoder layer with self-attn + cross-attn."""
    self_attn = DETRMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        block_prefix=f"{block_prefix}_self_attn",
        name=f"{block_prefix}_self_attn",
    )

    q = k = layers.Add(name=f"{block_prefix}_sa_qk_add")([x, query_pos])
    attn_output = self_attn(q, k, x)
    attn_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_sa_drop")(
        attn_output
    )
    x = layers.Add(name=f"{block_prefix}_sa_residual")([x, attn_output])
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_self_attn_layer_norm",
    )(x)

    cross_attn = DETRMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        block_prefix=f"{block_prefix}_encoder_attn",
        name=f"{block_prefix}_encoder_attn",
    )

    q_cross = layers.Add(name=f"{block_prefix}_ca_q_add")([x, query_pos])
    k_cross = layers.Add(name=f"{block_prefix}_ca_k_add")([memory, pos_embed])
    cross_output = cross_attn(q_cross, k_cross, memory)
    cross_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_ca_drop")(
        cross_output
    )
    x = layers.Add(name=f"{block_prefix}_ca_residual")([x, cross_output])
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_encoder_attn_layer_norm",
    )(x)

    ff_output = layers.Dense(
        dim_feedforward,
        activation="relu",
        name=f"{block_prefix}_fc1",
    )(x)
    ff_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_ff_drop")(ff_output)
    ff_output = layers.Dense(
        hidden_dim,
        name=f"{block_prefix}_fc2",
    )(ff_output)
    x = layers.Add(name=f"{block_prefix}_ff_residual")([x, ff_output])
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_final_layer_norm",
    )(x)

    return x


def detr_backbone(
    input_tensor,
    backbone_variant,
    include_normalization,
    normalization_mode,
    data_format="channels_last",
    channels_axis=-1,
):
    """Build a ResNet backbone (ResNet50 / ResNet101) for DETR.

    Layer naming mirrors HuggingFace's DETR backbone so weights can be
    transferred directly.
    """
    block_repeats = {
        "ResNet50": [3, 4, 6, 3],
        "ResNet101": [3, 4, 23, 3],
    }[backbone_variant]

    x = (
        ImageNormalizationLayer(mode=normalization_mode)(input_tensor)
        if include_normalization
        else input_tensor
    )

    x = layers.ZeroPadding2D(padding=3, data_format=data_format)(x)
    x = layers.Conv2D(
        64,
        7,
        strides=2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="backbone_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name="backbone_bn1",
    )(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
    x = layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="valid",
        data_format=data_format,
    )(x)

    filters_list = [64, 128, 256, 512]

    for stage_idx, num_blocks in enumerate(block_repeats):
        filters = filters_list[stage_idx]
        for block_idx in range(num_blocks):
            prefix = f"backbone_layer{stage_idx + 1}_{block_idx}"
            strides = 2 if block_idx == 0 and stage_idx > 0 else 1
            residual = x

            x = layers.Conv2D(
                filters,
                1,
                strides=1,
                padding="valid",
                use_bias=False,
                data_format=data_format,
                name=f"{prefix}_conv1",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"{prefix}_bn1",
            )(x)
            x = layers.ReLU()(x)

            if strides > 1:
                x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
                x = layers.Conv2D(
                    filters,
                    3,
                    strides=strides,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_conv2",
                )(x)
            else:
                x = layers.Conv2D(
                    filters,
                    3,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_conv2",
                )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"{prefix}_bn2",
            )(x)
            x = layers.ReLU()(x)

            x = layers.Conv2D(
                filters * 4,
                1,
                strides=1,
                padding="valid",
                use_bias=False,
                data_format=data_format,
                name=f"{prefix}_conv3",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"{prefix}_bn3",
            )(x)

            in_channels = residual.shape[channels_axis]
            out_channels = filters * 4
            if strides != 1 or in_channels != out_channels:
                if strides > 1:
                    residual = layers.ZeroPadding2D(padding=0, data_format=data_format)(
                        residual
                    )
                residual = layers.Conv2D(
                    out_channels,
                    1,
                    strides=strides,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_downsample_conv",
                )(residual)
                residual = layers.BatchNormalization(
                    axis=channels_axis,
                    epsilon=1e-5,
                    momentum=0.1,
                    name=f"{prefix}_downsample_bn",
                )(residual)

            x = layers.Add()([x, residual])
            x = layers.ReLU()(x)

    return x


def detr_encoder(
    backbone_features,
    hidden_dim,
    num_heads,
    num_encoder_layers,
    dim_feedforward,
    dropout_rate,
):
    """Build DETR's transformer encoder on top of backbone features.

    Projects the backbone's ``(B, H, W, 2048)`` feature map down to
    ``hidden_dim`` channels with a 1x1 conv, adds sinusoidal 2-D
    position embeddings, flattens both the features and the positions
    into ``(B, H*W, hidden_dim)`` token sequences, and runs
    ``num_encoder_layers`` post-norm transformer encoder layers
    (self-attention with positional embeddings added to Q/K, then FFN).

    Args:
        backbone_features: ResNet backbone output, ``(B, H/32, W/32, C)``
            for ``channels_last`` (C=2048 for ResNet-50).
        hidden_dim: Transformer model dimension.
        num_heads: Number of self-attention heads.
        num_encoder_layers: Number of stacked encoder layers.
        dim_feedforward: FFN dimension inside each encoder layer.
        dropout_rate: Dropout probability inside attention/FFN.

    Returns:
        encoder_output: ``(B, H*W, hidden_dim)`` encoded token sequence.
        pos: ``(B, H*W, hidden_dim)`` flattened position embeddings,
            reused by the decoder's cross-attention.
    """
    data_format = keras.config.image_data_format()

    projected = layers.Conv2D(
        hidden_dim,
        1,
        padding="valid",
        data_format=data_format,
        name="input_projection",
    )(backbone_features)

    pos_embed = DETRPositionEmbeddingSine(
        hidden_dim=hidden_dim,
        name="position_embedding",
    )(projected)

    src = DETRFlattenFeatures(hidden_dim, name="flatten_src")(projected)
    pos = DETRFlattenFeatures(hidden_dim, name="flatten_pos")(pos_embed)

    encoder_output = src
    for i in range(num_encoder_layers):
        encoder_output = detr_encoder_layer(
            encoder_output,
            pos,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            block_prefix=f"encoder_layers_{i}",
        )

    return encoder_output, pos


def detr_decoder(
    encoder_output,
    pos,
    hidden_dim,
    num_heads,
    num_decoder_layers,
    dim_feedforward,
    dropout_rate,
    num_queries,
):
    """Build DETR's transformer decoder on top of encoder outputs.

    Creates ``num_queries`` learned query position embeddings, runs
    ``num_decoder_layers`` post-norm transformer decoder layers
    (self-attention between queries with query positions added to Q/K,
    then cross-attention to the encoder memory with image positions
    added to keys, then FFN), and applies a final LayerNorm. Each
    decoder layer starts from zeros and is offset by the learned
    queries; the final hidden state is what classification + bbox
    heads consume in ``DETRDetect``.

    Args:
        encoder_output: Encoded token sequence from :func:`detr_encoder`.
        pos: Flattened image position embeddings (also from
            :func:`detr_encoder`); added to encoder keys in cross-attention.
        hidden_dim: Transformer model dimension.
        num_heads: Number of attention heads.
        num_decoder_layers: Number of stacked decoder layers.
        dim_feedforward: FFN dimension inside each decoder layer.
        dropout_rate: Dropout probability inside attention/FFN.
        num_queries: Number of learned object queries.

    Returns:
        Decoder ``last_hidden_state`` of shape
        ``(B, num_queries, hidden_dim)`` — the DETR equivalent of
        HuggingFace's ``DetrModel.last_hidden_state``.
    """
    query_embed = DETRExpandQueryEmbedding(
        num_queries,
        hidden_dim,
        name="query_position_embeddings",
    )(encoder_output)

    decoder_output = ops.zeros_like(query_embed)
    for i in range(num_decoder_layers):
        decoder_output = detr_decoder_layer(
            decoder_output,
            encoder_output,
            pos,
            query_embed,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            block_prefix=f"decoder_layers_{i}",
        )

    last_hidden_state = layers.LayerNormalization(
        epsilon=1e-5,
        name="decoder_layernorm",
    )(decoder_output)

    return last_hidden_state


def detr_functional(
    inputs,
    backbone_variant,
    hidden_dim,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    dim_feedforward,
    dropout_rate,
    num_queries,
    include_normalization,
    normalization_mode,
):
    """Build the full DETR architecture from an input tensor (no class heads).

    Top-level orchestrator that wires the three architectural stages:

    1. :func:`detr_backbone` — ResNet-50 / ResNet-101 produces a
       ``(B, H/32, W/32, 2048)`` feature map.
    2. :func:`detr_encoder` — 1x1 input projection + sine position
       embedding + flatten + ``num_encoder_layers`` transformer encoder
       layers.
    3. :func:`detr_decoder` — learned object queries +
       ``num_decoder_layers`` transformer decoder layers + final
       LayerNorm.

    Classification + bounding-box prediction heads are intentionally
    not built here — they are added by :class:`DETRDetect`, which
    composes :class:`DetrModel` around this graph.

    Args:
        inputs: Keras input tensor of shape ``(B, H, W, 3)`` (or
            ``(B, 3, H, W)`` for ``channels_first``).
        backbone_variant: ``"ResNet50"`` or ``"ResNet101"``.
        hidden_dim: Transformer model dimension.
        num_heads: Number of attention heads in encoder and decoder.
        num_encoder_layers: Number of transformer encoder layers.
        num_decoder_layers: Number of transformer decoder layers.
        dim_feedforward: FFN dimension inside each transformer layer.
        dropout_rate: Dropout probability inside attention/FFN.
        num_queries: Number of learned object queries.
        include_normalization: Whether to prepend an
            :class:`ImageNormalizationLayer` (apply only when ``inputs``
            is in raw ``[0, 255]`` pixel space).
        normalization_mode: Normalization preset (e.g. ``"imagenet"``).

    Returns:
        Decoder ``last_hidden_state`` of shape
        ``(B, num_queries, hidden_dim)``.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    backbone_features = detr_backbone(
        inputs,
        backbone_variant=backbone_variant,
        include_normalization=include_normalization,
        normalization_mode=normalization_mode,
        data_format=data_format,
        channels_axis=channels_axis,
    )
    encoder_output, pos = detr_encoder(
        backbone_features,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
    )
    return detr_decoder(
        encoder_output,
        pos,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
        num_queries=num_queries,
    )


@keras.saving.register_keras_serializable(package="kmodels")
class DetrModel(BaseModel):
    """DETR backbone + transformer encoder/decoder (no detection heads).

    Matches the HuggingFace ``DetrModel`` pattern — outputs the decoder
    ``last_hidden_state`` with shape ``(B, num_queries, hidden_dim)``.
    Wraps the functional graph built by :func:`detr_functional`: a
    ResNet-50/101 backbone, a stack of post-norm transformer encoder
    layers with sine 2D position embeddings, and a stack of post-norm
    transformer decoder layers with learned object queries plus a
    final LayerNorm. Classification and bbox prediction heads are
    intentionally pruned from the output graph; use
    :class:`DETRDetect` if you want full detection outputs.

    Reference:
        - `End-to-End Object Detection with Transformers
          <https://arxiv.org/abs/2005.12872>`_

    Args:
        backbone_variant: Backbone architecture. One of ``"ResNet50"``
            or ``"ResNet101"``. Defaults to ``"ResNet50"``.
        hidden_dim: Transformer model dimension (channel width of both
            encoder and decoder, and of the input projection that
            reduces the backbone's 2048-channel feature map).
            Defaults to ``256``.
        num_heads: Number of attention heads in every transformer
            self-attention and cross-attention layer.
            Defaults to ``8``.
        num_encoder_layers: Number of stacked transformer encoder
            layers. Defaults to ``6``.
        num_decoder_layers: Number of stacked transformer decoder
            layers. Defaults to ``6``.
        dim_feedforward: FFN intermediate dimension inside each
            encoder / decoder layer. Defaults to ``2048``.
        dropout_rate: Dropout probability used in attention and FFN
            sub-layers. Defaults to ``0.1``.
        num_queries: Number of learned object queries — also the
            number of detections produced per image.
            Defaults to ``100``.
        include_normalization: If ``True``, prepend an
            :class:`ImageNormalizationLayer` to the backbone so the
            model accepts raw ``[0, 255]`` pixel values. Set to
            ``False`` when inputs are already pre-normalized.
            Defaults to ``True``.
        normalization_mode: Normalization preset passed to the
            built-in normalization layer (e.g. ``"imagenet"``).
            Ignored when ``include_normalization=False``.
            Defaults to ``"imagenet"``.
        input_shape: Image input shape excluding the batch axis. When
            ``None``, defaults to ``(800, 800, 3)``.
        input_tensor: Optional pre-existing Keras tensor to use as the
            model input instead of creating a new :class:`Input`.
            Defaults to ``None``.
        name: Model name. Defaults to ``"DetrModel"``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`BaseModel` / :class:`keras.Model`.
    """

    BASE_MODEL_CONFIG = DETR_CONFIG
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "detr"

    def __init__(
        self,
        backbone_variant="ResNet50",
        hidden_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout_rate=0.1,
        num_queries=100,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="DetrModel",
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (800, 800, 3)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        last_hidden_state = detr_functional(
            img_input,
            backbone_variant=backbone_variant,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            num_queries=num_queries,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
        )

        super().__init__(
            inputs=img_input, outputs=last_hidden_state, name=name, **kwargs
        )

        self.backbone_variant = backbone_variant
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.num_queries = num_queries
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_variant": self.backbone_variant,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "num_queries": self.num_queries,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def config_from_hf(cls, hf_config):
        backbone = hf_config.get("backbone", "resnet50") or "resnet50"
        backbone_variant = "ResNet101" if "101" in backbone else "ResNet50"
        return {
            "backbone_variant": backbone_variant,
            "hidden_dim": hf_config["d_model"],
            "num_heads": hf_config["encoder_attention_heads"],
            "num_encoder_layers": hf_config["encoder_layers"],
            "num_decoder_layers": hf_config["decoder_layers"],
            "dim_feedforward": hf_config["encoder_ffn_dim"],
            "dropout_rate": hf_config["dropout"],
            "num_queries": hf_config["num_queries"],
            "include_normalization": False,
        }


@keras.saving.register_keras_serializable(package="kmodels")
class DETRDetect(BaseModel):
    """DETR object detection model (encoder-decoder transformer + heads).

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Loads pretrained weights via ``DETRDetect.from_weights(...)``.
    See ``BaseModel.from_weights`` for the loading API.
    """

    BASE_MODEL_CONFIG = DETR_CONFIG
    BASE_WEIGHT_CONFIG = DETR_WEIGHTS
    HF_MODEL_TYPE = "detr"

    def __init__(
        self,
        backbone_variant="ResNet50",
        hidden_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout_rate=0.1,
        num_queries=100,
        num_classes=92,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="DETRDetect",
        **kwargs,
    ):
        base = DetrModel(
            backbone_variant=backbone_variant,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            num_queries=num_queries,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_model",
        )
        last_hidden_state = base.output

        logits = layers.Dense(
            num_classes,
            name="class_labels_classifier",
        )(last_hidden_state)

        bbox = layers.Dense(hidden_dim, activation="relu", name="bbox_predictor_0")(
            last_hidden_state
        )
        bbox = layers.Dense(hidden_dim, activation="relu", name="bbox_predictor_1")(
            bbox
        )
        bbox = layers.Dense(4, name="bbox_predictor_2")(bbox)
        bbox = layers.Activation("sigmoid", name="bbox_sigmoid")(bbox)

        outputs = {"logits": logits, "pred_boxes": bbox}

        super().__init__(inputs=base.input, outputs=outputs, name=name, **kwargs)

        self.backbone_variant = backbone_variant
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_variant": self.backbone_variant,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def config_from_hf(cls, hf_config):
        backbone = hf_config.get("backbone", "resnet50") or "resnet50"
        backbone_variant = "ResNet101" if "101" in backbone else "ResNet50"
        return {
            "backbone_variant": backbone_variant,
            "hidden_dim": hf_config["d_model"],
            "num_heads": hf_config["encoder_attention_heads"],
            "num_encoder_layers": hf_config["encoder_layers"],
            "num_decoder_layers": hf_config["decoder_layers"],
            "dim_feedforward": hf_config["encoder_ffn_dim"],
            "dropout_rate": hf_config["dropout"],
            "num_queries": hf_config["num_queries"],
            "num_classes": hf_num_labels(hf_config) + 1,
            "include_normalization": False,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        transfer_detr_weights(keras_model, hf_state_dict)
