import keras
from keras import layers, ops, utils

from kerasformers.base import BaseModel
from kerasformers.base.base_model import hf_num_classes
from kerasformers.models.detr.detr_layers import (
    DETRExpandQueryEmbedding,
    DETRFlattenFeatures,
    DETRMaskHeadSmallConv,
    DETRMHAttentionMap,
    DETRMultiHeadAttention,
    DETRPositionEmbeddingSine,
)
from kerasformers.utils import standardize_input_shape

from .config import (
    DETR_CONFIG,
    DETR_SEGMENT_CONFIG,
    DETR_SEGMENT_WEIGHTS,
    DETR_WEIGHTS,
)


def detr_encoder_layer(
    x,
    pos_embed,
    hidden_dim,
    num_heads,
    dim_feedforward,
    dropout_rate=0.1,
    block_prefix="encoder_layers_0",
):
    """One post-LN DETR transformer encoder layer (self-attn → FFN).

    Mirrors the canonical FB-Research DETR encoder block:

    1. ``q = k = x + pos_embed`` (positional encoding is added to the
       query/key paths only — the value stream stays unchanged), then
       self-attention.
    2. Dropout → residual add → LayerNorm.
    3. Feed-forward: ``Linear(hidden→ff) → ReLU → Dropout →
       Linear(ff→hidden)``.
    4. Residual add → LayerNorm.

    All sublayer names are deterministic (``{block_prefix}_*``) so the
    PyTorch state-dict can be transferred by name during checkpoint
    conversion.

    Reference:
        - `End-to-End Object Detection with Transformers
          <https://arxiv.org/abs/2005.12872>`_

    Args:
        x: Flattened image-feature token sequence of shape
            ``(B, H*W, hidden_dim)``.
        pos_embed: Sine positional embedding of shape
            ``(B, H*W, hidden_dim)``, added to the query/key paths of
            the self-attention.
        hidden_dim: Model / token dimension (DETR uses 256).
        num_heads: Number of attention heads. ``hidden_dim`` must be
            divisible by it.
        dim_feedforward: Hidden dimension of the FFN's intermediate
            Dense layer (DETR uses 2048).
        dropout_rate: Dropout applied to the attention output and the
            FFN's intermediate activations.
        block_prefix: Prefix used to name every sublayer in this block.

    Returns:
        Tensor of shape ``(B, H*W, hidden_dim)`` — the encoder layer's
        output.
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
    """One post-LN DETR decoder layer (self-attn → cross-attn → FFN).

    Mirrors the canonical FB-Research DETR decoder block:

    1. **Self-attn over queries**: ``q = k = x + query_pos`` (learned
       object-query positional embedding added to Q/K only), ``v = x``;
       then Dropout → residual → LayerNorm.
    2. **Cross-attn into encoder memory**: ``q = x + query_pos``,
       ``k = memory + pos_embed`` (sine positional encoding on the
       image side), ``v = memory``; then Dropout → residual → LayerNorm.
    3. **Feed-forward**: ``Linear(hidden→ff) → ReLU → Dropout →
       Linear(ff→hidden)``; then residual → LayerNorm.

    Sublayer names are deterministic (``{block_prefix}_*``) so the
    PyTorch state-dict can be transferred by name during conversion.

    Reference:
        - `End-to-End Object Detection with Transformers
          <https://arxiv.org/abs/2005.12872>`_

    Args:
        x: Current decoder token sequence of shape
            ``(B, num_queries, hidden_dim)``. Starts at zero in the
            first decoder layer.
        memory: Encoder output of shape ``(B, H*W, hidden_dim)``, used
            as keys and values in the cross-attention.
        pos_embed: Sine positional embedding of shape
            ``(B, H*W, hidden_dim)``, added to the cross-attention's
            key path so the encoder side keeps its spatial geometry.
        query_pos: Learned object-query embedding of shape
            ``(B, num_queries, hidden_dim)``, added to the Q/K path of
            self-attention and to the Q path of cross-attention.
        hidden_dim: Model / token dimension (DETR uses 256).
        num_heads: Number of attention heads. ``hidden_dim`` must be
            divisible by it.
        dim_feedforward: Hidden dimension of the FFN's intermediate
            Dense layer (DETR uses 2048).
        dropout_rate: Dropout applied to each attention output and to
            the FFN's intermediate activations.
        block_prefix: Prefix used to name every sublayer in this block.

    Returns:
        Tensor of shape ``(B, num_queries, hidden_dim)`` — the decoder
        layer's output, ready to feed the next decoder layer or the
        detection head.
    """
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
    data_format="channels_last",
    channels_axis=-1,
):
    """ResNet-50 / ResNet-101 backbone used by DETR.

    Standard torchvision-style ResNet (7×7 stem → max-pool → 4 stages
    of bottleneck residual blocks). Returns the **four** stage outputs
    (C2/C3/C4/C5 at strides 4/8/16/32) so downstream heads can do
    FPN-style fusion. The DETR encoder uses only the last (C5);
    :class:`DETRSegment`'s mask head uses C2/C3/C4 as well.

    Sublayer names follow the reference DETR backbone naming
    (``backbone_conv1``, ``backbone_layer{stage}_{block}_*``,
    ``*_downsample_*``), so :func:`transfer_detr_weights` can map the
    PyTorch state-dict directly without renaming.

    Args:
        input_tensor: Input image tensor — ``(B, H, W, 3)`` for
            ``channels_last`` or ``(B, 3, H, W)`` for ``channels_first``.
            Expected to be pre-normalized by
            :class:`DETRImageProcessor` (ImageNet mean/std).
        backbone_variant: ``"ResNet50"`` (block repeats 3-4-6-3) or
            ``"ResNet101"`` (3-4-23-3).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis matching ``data_format``
            (``-1`` for ``channels_last``, ``1`` for ``channels_first``).
            Used to align ``BatchNormalization`` axes.

    Returns:
        Tuple ``(c2, c3, c4, c5)`` — feature maps after stages 1..4 at
        strides 4, 8, 16, 32 with channel widths 256, 512, 1024, 2048.
        For ``channels_last`` shapes are ``(B, H/stride, W/stride, C)``;
        for ``channels_first`` they are ``(B, C, H/stride, W/stride)``.
    """
    depths = {
        "ResNet50": [3, 4, 6, 3],
        "ResNet101": [3, 4, 23, 3],
    }[backbone_variant]

    x = input_tensor

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
    stage_outputs = []

    for stage_idx, depths in enumerate(depths):
        filters = filters_list[stage_idx]
        for block_idx in range(depths):
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
        stage_outputs.append(x)

    return tuple(stage_outputs)


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
        projected: ``(B, H/32, W/32, hidden_dim)`` pre-encoder spatial
            map (the 1×1 ``input_projection`` output). Used by
            :class:`DETRSegment` as the mask head's ``features`` input.
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

    return encoder_output, pos, projected


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
        the reference ``DetrModel`` last hidden state.
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
    return_intermediates=False,
):
    """Build the full DETR architecture from an input tensor (no class heads).

    Top-level orchestrator that wires the three architectural stages:

    1. :func:`detr_backbone` — ResNet-50 / ResNet-101 produces multi-scale
       feature maps; the C5 (stride-32) map feeds the transformer.
    2. :func:`detr_encoder` — 1x1 input projection + sine position
       embedding + flatten + ``num_encoder_layers`` transformer encoder
       layers.
    3. :func:`detr_decoder` — learned object queries +
       ``num_decoder_layers`` transformer decoder layers + final
       LayerNorm.

    Classification + bounding-box prediction heads are intentionally
    not built here — they are added by :class:`DETRDetect`, which
    composes :class:`DetrModel` around this graph.
    :class:`DETRSegment` uses ``return_intermediates=True`` to also
    grab the multi-scale backbone features and the encoder output for
    its mask head.

    Args:
        inputs: Keras input tensor of shape ``(B, H, W, 3)`` (or
            ``(B, 3, H, W)`` for ``channels_first``). Expected to be
            pre-normalized by :class:`DETRImageProcessor`.
        backbone_variant: ``"ResNet50"`` or ``"ResNet101"``.
        hidden_dim: Transformer model dimension.
        num_heads: Number of attention heads in encoder and decoder.
        num_encoder_layers: Number of transformer encoder layers.
        num_decoder_layers: Number of transformer decoder layers.
        dim_feedforward: FFN dimension inside each transformer layer.
        dropout_rate: Dropout probability inside attention/FFN.
        num_queries: Number of learned object queries.
        return_intermediates: If ``True``, return a dict with keys
            ``"last_hidden_state"``, ``"encoder_output"``,
            ``"backbone_features"`` (tuple of C2/C3/C4/C5). If
            ``False`` (default), return only the decoder
            ``last_hidden_state``.

    Returns:
        If ``return_intermediates=False``: decoder
        ``last_hidden_state`` of shape ``(B, num_queries, hidden_dim)``.

        If ``return_intermediates=True``: dict with
        ``last_hidden_state`` ``(B, num_queries, hidden_dim)``,
        ``encoder_output`` ``(B, H*W, hidden_dim)``, and
        ``backbone_features`` (tuple of 4 stage feature maps).
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    backbone_features = detr_backbone(
        inputs,
        backbone_variant=backbone_variant,
        data_format=data_format,
        channels_axis=channels_axis,
    )
    encoder_output, pos, projected_features = detr_encoder(
        backbone_features[-1],
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
    )
    last_hidden_state = detr_decoder(
        encoder_output,
        pos,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
        num_queries=num_queries,
    )
    if return_intermediates:
        return {
            "last_hidden_state": last_hidden_state,
            "encoder_output": encoder_output,
            "projected_features": projected_features,
            "backbone_features": backbone_features,
        }
    return last_hidden_state


@keras.saving.register_keras_serializable(package="kerasformers")
class DetrModel(BaseModel):
    """DETR backbone + transformer encoder/decoder (no detection heads).

    Matches the reference ``DetrModel`` pattern — outputs the decoder
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
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()`` —
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `800`.
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
        image_size=800,
        input_tensor=None,
        name="DetrModel",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=image_size)
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
        self.image_size = image_size
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
                "image_size": self.image_size,
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
        }


@keras.saving.register_keras_serializable(package="kerasformers")
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
        image_size=800,
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
            image_size=image_size,
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
        self.image_size = base.image_size
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
                "image_size": self.image_size,
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
            "num_classes": hf_num_classes(hf_config) + 1,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_detr_hf_to_keras import transfer_detr_weights

        transfer_detr_weights(keras_model, hf_state_dict)


@keras.saving.register_keras_serializable(package="kerasformers")
class DETRSegment(BaseModel):
    """DETR for panoptic / instance segmentation — detection + per-query masks.

    Composes the
    detection model (class + bbox heads identical to
    :class:`DETRDetect`) and adds the segmentation head — a multi-head
    attention map between decoder queries and encoder features
    (:class:`DETRMHAttentionMap`) plus a small FPN-style mask CNN
    (:class:`DETRMaskHeadSmallConv`) that fuses the attention map with
    multi-scale backbone features (C2 / C3 / C4 at strides 4 / 8 / 16)
    through three nearest-neighbour upsampling stages.

    Output dict:

    .. code-block:: python

        out = model(images)
        out["logits"]      # (B, num_queries, num_classes) — class logits
        out["pred_boxes"]  # (B, num_queries, 4) — sigmoid cxcywh in [0, 1]
        out["pred_masks"]  # (B, num_queries, H/4, W/4) — mask logits

    Construction:

    >>> DETRSegment.from_weights("hf:facebook/detr-resnet-50-panoptic")

    Reference:
        - `End-to-End Object Detection with Transformers
          <https://arxiv.org/abs/2005.12872>`_ (section 5 covers
          panoptic segmentation).

    Args:
        backbone_variant: ``"ResNet50"`` or ``"ResNet101"``.
            Defaults to ``"ResNet50"``.
        hidden_dim: Transformer model dimension. Defaults to ``256``.
        num_heads: Attention head count in the transformer (also the
            head count of :class:`DETRMHAttentionMap`). Defaults to ``8``.
        num_encoder_layers: Number of transformer encoder layers.
            Defaults to ``6``.
        num_decoder_layers: Number of transformer decoder layers.
            Defaults to ``6``.
        dim_feedforward: FFN intermediate dimension. Defaults to ``2048``.
        dropout_rate: Attention / FFN dropout rate. Defaults to ``0.1``.
        num_queries: Number of learned object queries (= number of
            mask + class + bbox predictions per image).
            Defaults to ``100``.
        num_classes: Class-head output dim (panoptic checkpoints
            use ``250``). Defaults to ``250``.
        image_size: Input image specification. Defaults to ``800``.
        input_tensor: Optional pre-existing Keras tensor for the
            ``images`` input.
        name: Model name. Defaults to ``"DETRSegment"``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`BaseModel`.
    """

    BASE_MODEL_CONFIG = DETR_SEGMENT_CONFIG
    BASE_WEIGHT_CONFIG = DETR_SEGMENT_WEIGHTS
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
        num_classes=250,
        image_size=800,
        input_tensor=None,
        name="DETRSegment",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if data_format == "channels_first":
            image_h = image_size[1]
            image_w = image_size[2]
        else:
            image_h = image_size[0]
            image_w = image_size[1]
        h32 = image_h // 32
        w32 = image_w // 32

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=image_size)
            else:
                img_input = input_tensor

        intermediates = detr_functional(
            img_input,
            backbone_variant=backbone_variant,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            num_queries=num_queries,
            return_intermediates=True,
        )
        last_hidden_state = intermediates["last_hidden_state"]
        encoder_output = intermediates["encoder_output"]
        projected_features = intermediates["projected_features"]
        backbone_features = intermediates["backbone_features"]
        c2, c3, c4, _ = backbone_features

        logits = layers.Dense(num_classes, name="class_labels_classifier")(
            last_hidden_state
        )
        bbox = layers.Dense(hidden_dim, activation="relu", name="bbox_predictor_0")(
            last_hidden_state
        )
        bbox = layers.Dense(hidden_dim, activation="relu", name="bbox_predictor_1")(
            bbox
        )
        bbox = layers.Dense(4, name="bbox_predictor_2")(bbox)
        bbox = layers.Activation("sigmoid", name="bbox_sigmoid")(bbox)

        memory = layers.Reshape((h32, w32, hidden_dim), name="memory_reshape")(
            encoder_output
        )
        if data_format == "channels_first":
            c2_cl = layers.Permute((2, 3, 1), name="c2_to_channels_last")(c2)
            c3_cl = layers.Permute((2, 3, 1), name="c3_to_channels_last")(c3)
            c4_cl = layers.Permute((2, 3, 1), name="c4_to_channels_last")(c4)
            features_cl = layers.Permute((2, 3, 1), name="projected_to_channels_last")(
                projected_features
            )
        else:
            c2_cl, c3_cl, c4_cl, features_cl = c2, c3, c4, projected_features

        bbox_mask = DETRMHAttentionMap(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            name="bbox_attention",
        )(last_hidden_state, memory)

        fpn_dims = (c4_cl.shape[-1], c3_cl.shape[-1], c2_cl.shape[-1])
        seg_masks = DETRMaskHeadSmallConv(
            dim=hidden_dim + num_heads,
            fpn_dims=fpn_dims,
            context_dim=hidden_dim,
            name="mask_head",
        )(features_cl, bbox_mask, [c4_cl, c3_cl, c2_cl])

        outputs = {
            "logits": logits,
            "pred_boxes": bbox,
            "pred_masks": seg_masks,
        }

        super().__init__(inputs=img_input, outputs=outputs, name=name, **kwargs)

        self.backbone_variant = backbone_variant
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.image_size = image_size
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
                "image_size": self.image_size,
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
            "num_classes": hf_num_classes(hf_config) + 1,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_detr_hf_to_keras import transfer_detr_segment_weights

        transfer_detr_segment_weights(keras_model, hf_state_dict)
