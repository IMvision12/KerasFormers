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


def build_detr_backbone(
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


@keras.saving.register_keras_serializable(package="kmodels")
class DETRDetect(BaseModel):
    """DETR object detection model (encoder-decoder transformer + heads).

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Loads pretrained weights via ``DETRDetect.from_weights(...)``.
    See ``BaseModel.from_weights`` for the loading API.
    """

    KMODELS_CONFIG = DETR_CONFIG
    KMODELS_WEIGHTS = DETR_WEIGHTS

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
        if input_shape is None:
            input_shape = (800, 800, 3)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        inputs = img_input

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        backbone_features = build_detr_backbone(
            inputs,
            backbone_variant=backbone_variant,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            data_format=data_format,
            channels_axis=channels_axis,
        )

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

        query_embed_layer = DETRExpandQueryEmbedding(
            num_queries,
            hidden_dim,
            name="query_position_embeddings",
        )
        query_embed = query_embed_layer(encoder_output)

        decoder_input = ops.zeros_like(query_embed)

        decoder_output = decoder_input
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

        decoder_output = layers.LayerNormalization(
            epsilon=1e-5,
            name="decoder_layernorm",
        )(decoder_output)

        logits = layers.Dense(
            num_classes,
            name="class_labels_classifier",
        )(decoder_output)

        bbox = layers.Dense(hidden_dim, activation="relu", name="bbox_predictor_0")(
            decoder_output
        )
        bbox = layers.Dense(hidden_dim, activation="relu", name="bbox_predictor_1")(
            bbox
        )
        bbox = layers.Dense(4, name="bbox_predictor_2")(bbox)
        bbox = layers.Activation("sigmoid", name="bbox_sigmoid")(bbox)

        outputs = {"logits": logits, "pred_boxes": bbox}

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

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
    def _config_from_hf(cls, hf_config):
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
    def _transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_detr_torch_to_keras import transfer_detr_weights

        transfer_detr_weights(keras_model, hf_state_dict)
