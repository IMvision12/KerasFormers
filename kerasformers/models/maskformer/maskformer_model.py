import keras
from keras import layers, ops

from kerasformers.base import BaseModel
from kerasformers.utils import standardize_input_shape

from .config import MASKFORMER_CONFIG, MASKFORMER_WEIGHTS
from .maskformer_layers import (
    MaskFormerDetrAttention,
    MaskFormerExpandQueryEmbedding,
    MaskFormerSinePositionEmbedding,
)
from .maskformer_swin_layers import MaskFormerSwinBackbone


def maskformer_fpn_stem(
    features,
    fpn_feature_size,
    data_format,
    channels_axis,
    block_prefix="pixel_decoder_fpn_stem",
):
    """Initial FPN stem: 3×3 conv (no bias) + GroupNorm + ReLU on the coarsest backbone feature."""
    x = layers.Conv2D(
        fpn_feature_size,
        3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{block_prefix}_conv",
    )(features)
    x = layers.GroupNormalization(
        groups=32, axis=channels_axis, epsilon=1e-5, name=f"{block_prefix}_norm"
    )(x)
    x = layers.Activation("relu", name=f"{block_prefix}_relu")(x)
    return x


def maskformer_fpn_layer(
    x,
    skip,
    fpn_feature_size,
    data_format,
    channels_axis,
    block_prefix="pixel_decoder_fpn_layer_0",
):
    """One FPN stage: lateral 1×1 proj of ``skip`` + 2× upsample of ``x`` + 3×3 output conv (no bias)."""
    lateral = layers.Conv2D(
        fpn_feature_size,
        1,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name=f"{block_prefix}_proj_conv",
    )(skip)
    lateral = layers.GroupNormalization(
        groups=32, axis=channels_axis, epsilon=1e-5, name=f"{block_prefix}_proj_norm"
    )(lateral)

    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="nearest",
        data_format=data_format,
        name=f"{block_prefix}_upsample",
    )(x)
    x = layers.Add(name=f"{block_prefix}_add")([x, lateral])

    x = layers.Conv2D(
        fpn_feature_size,
        3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{block_prefix}_block_conv",
    )(x)
    x = layers.GroupNormalization(
        groups=32, axis=channels_axis, epsilon=1e-5, name=f"{block_prefix}_block_norm"
    )(x)
    x = layers.Activation("relu", name=f"{block_prefix}_block_relu")(x)
    return x


def maskformer_pixel_decoder(
    features, fpn_feature_size, mask_feature_size, data_format, channels_axis
):
    """FPN-style pixel decoder.

    ``features`` is the ordered list of 4 backbone feature maps from
    finest to coarsest. The FPN starts at the coarsest and progressively
    fuses laterals while upsampling 2× per step.

    Returns:
        mask_features: ``(B, H/4, W/4, mask_feature_size)``.
    """
    x = maskformer_fpn_stem(features[-1], fpn_feature_size, data_format, channels_axis)
    for i, skip in enumerate(reversed(features[:-1])):
        x = maskformer_fpn_layer(
            x,
            skip,
            fpn_feature_size,
            data_format,
            channels_axis,
            block_prefix=f"pixel_decoder_fpn_layer_{i}",
        )
    mask_features = layers.Conv2D(
        mask_feature_size,
        3,
        padding="same",
        use_bias=True,
        data_format=data_format,
        name="pixel_decoder_mask_projection",
    )(x)
    return mask_features


def maskformer_decoder_layer(
    hidden_states,
    memory,
    memory_pos,
    query_pos,
    hidden_dim,
    num_heads,
    ffn_dim,
    dropout_rate=0.0,
    block_prefix="transformer_decoder_layers_0",
):
    """One MaskFormer / DETR decoder layer.

    Order: self-attn → post-LN → cross-attn (into ``memory``) → post-LN
    → FFN → post-LN. Query positional embedding (``query_pos``) is added
    to the Q and K paths of self-attention and to the Q path of
    cross-attention. Image positional embedding (``memory_pos``) is
    added to the K path of cross-attention.

    Mirrors HF ``DetrDecoderLayer`` sublayer naming
    (``self_attn``, ``self_attn_layer_norm``, ``encoder_attn``,
    ``encoder_attn_layer_norm``, ``fc1``, ``fc2``, ``final_layer_norm``).
    """
    residual = hidden_states
    q = k = layers.Add(name=f"{block_prefix}_sa_qk_add")([hidden_states, query_pos])
    attn_out = MaskFormerDetrAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        name=f"{block_prefix}_self_attn",
    )(q, k, hidden_states)
    hidden_states = layers.Add(name=f"{block_prefix}_sa_residual")([residual, attn_out])
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_self_attn_layer_norm"
    )(hidden_states)

    residual = hidden_states
    q_cross = layers.Add(name=f"{block_prefix}_ca_q_add")([hidden_states, query_pos])
    k_cross = layers.Add(name=f"{block_prefix}_ca_k_add")([memory, memory_pos])
    cross_out = MaskFormerDetrAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        name=f"{block_prefix}_encoder_attn",
    )(q_cross, k_cross, memory)
    hidden_states = layers.Add(name=f"{block_prefix}_ca_residual")(
        [residual, cross_out]
    )
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_encoder_attn_layer_norm"
    )(hidden_states)

    residual = hidden_states
    y = layers.Dense(ffn_dim, name=f"{block_prefix}_fc1")(hidden_states)
    y = layers.Activation("relu", name=f"{block_prefix}_fc1_relu")(y)
    y = layers.Dense(hidden_dim, name=f"{block_prefix}_fc2")(y)
    hidden_states = layers.Add(name=f"{block_prefix}_ff_residual")([residual, y])
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_final_layer_norm"
    )(hidden_states)
    return hidden_states


def maskformer_transformer_decoder(
    memory_feature,
    hidden_dim,
    num_layers,
    num_heads,
    ffn_dim,
    num_queries,
    data_format,
    dropout_rate=0.0,
):
    """MaskFormer transformer decoder.

    Inputs the projected coarsest backbone feature (post-input-projection
    1×1 conv) as the cross-attention memory, builds a 2D sine position
    embedding for it, runs ``num_layers`` DETR-style decoder layers with
    learned object queries, and applies a final LayerNorm.

    Returns the final decoder hidden state of shape
    ``(B, num_queries, hidden_dim)``.
    """
    memory_pos_2d = MaskFormerSinePositionEmbedding(
        hidden_dim=hidden_dim,
        data_format=data_format,
        name="transformer_decoder_position_embedding",
    )(memory_feature)

    b_ref = memory_feature
    mem = memory_feature
    pos = memory_pos_2d
    if data_format == "channels_first":
        mem = ops.transpose(mem, (0, 2, 3, 1))
        pos = ops.transpose(pos, (0, 2, 3, 1))
    memory_flat = layers.Reshape(
        (-1, hidden_dim), name="transformer_decoder_flatten_mem"
    )(mem)
    memory_pos_flat = layers.Reshape(
        (-1, hidden_dim), name="transformer_decoder_flatten_pos"
    )(pos)

    query_pos = MaskFormerExpandQueryEmbedding(
        num_queries=num_queries,
        hidden_dim=hidden_dim,
        name="transformer_decoder_queries_embedder",
    )(b_ref)

    hidden_states = ops.zeros_like(query_pos)
    for i in range(num_layers):
        hidden_states = maskformer_decoder_layer(
            hidden_states,
            memory=memory_flat,
            memory_pos=memory_pos_flat,
            query_pos=query_pos,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout_rate=dropout_rate,
            block_prefix=f"transformer_decoder_layers_{i}",
        )

    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name="transformer_decoder_layernorm"
    )(hidden_states)
    return hidden_states


def maskformer_mask_embedder(hidden_states, hidden_dim, mask_feature_size):
    """3-layer per-query MLP producing mask embeddings.

    Sequence ``Dense → ReLU → Dense → ReLU → Dense`` with the last
    ``Dense`` projecting to ``mask_feature_size`` so the per-query
    mask embedding can be dotted with the pixel-decoder mask features.
    """
    x = layers.Dense(hidden_dim, name="mask_embedder_0")(hidden_states)
    x = layers.Activation("relu", name="mask_embedder_0_relu")(x)
    x = layers.Dense(hidden_dim, name="mask_embedder_1")(x)
    x = layers.Activation("relu", name="mask_embedder_1_relu")(x)
    x = layers.Dense(mask_feature_size, name="mask_embedder_2")(x)
    return x


def maskformer_functional(
    pixel_values,
    *,
    backbone_embed_dim,
    backbone_depths,
    backbone_num_heads,
    backbone_window_size,
    fpn_feature_size,
    mask_feature_size,
    decoder_d_model,
    decoder_layers,
    decoder_heads,
    decoder_ffn_dim,
    num_queries,
    num_labels,
    data_format,
    channels_axis,
):
    """Build the full MaskFormer graph (backbone + pixel decoder + transformer + heads)."""
    backbone = MaskFormerSwinBackbone(
        embed_dim=backbone_embed_dim,
        depths=backbone_depths,
        num_heads=backbone_num_heads,
        window_size=backbone_window_size,
        data_format=data_format,
        name="backbone",
    )
    backbone_features = backbone(pixel_values)

    mask_features = maskformer_pixel_decoder(
        backbone_features,
        fpn_feature_size=fpn_feature_size,
        mask_feature_size=mask_feature_size,
        data_format=data_format,
        channels_axis=channels_axis,
    )

    memory_feature = layers.Conv2D(
        decoder_d_model,
        1,
        padding="valid",
        use_bias=True,
        data_format=data_format,
        name="transformer_decoder_input_projection",
    )(backbone_features[-1])

    decoder_hidden = maskformer_transformer_decoder(
        memory_feature,
        hidden_dim=decoder_d_model,
        num_layers=decoder_layers,
        num_heads=decoder_heads,
        ffn_dim=decoder_ffn_dim,
        num_queries=num_queries,
        data_format=data_format,
    )

    class_logits = layers.Dense(num_labels + 1, name="class_predictor")(decoder_hidden)
    mask_embeddings = maskformer_mask_embedder(
        decoder_hidden, decoder_d_model, mask_feature_size
    )
    mask_eq = "bqc,bhwc->bqhw" if data_format == "channels_last" else "bqc,bchw->bqhw"
    mask_logits = ops.einsum(mask_eq, mask_embeddings, mask_features)

    return {
        "class_queries_logits": class_logits,
        "masks_queries_logits": mask_logits,
    }


@keras.saving.register_keras_serializable(package="kerasformers")
class MaskFormerModel(BaseModel):
    """MaskFormer base model (backbone + pixel decoder + transformer, no segment heads).

    Returns the decoder ``last_hidden_state`` along with the pixel decoder
    mask features and the per-stage backbone features so callers can add
    custom heads.

    Reference:
    - [Per-Pixel Classification is Not All You Need for Semantic
      Segmentation](https://arxiv.org/abs/2107.06278)
    """

    BASE_MODEL_CONFIG = MASKFORMER_CONFIG
    HF_MODEL_TYPE = "maskformer"

    def __init__(
        self,
        backbone_embed_dim=96,
        backbone_depths=(2, 2, 6, 2),
        backbone_num_heads=(3, 6, 12, 24),
        backbone_window_size=7,
        fpn_feature_size=256,
        mask_feature_size=256,
        decoder_d_model=256,
        decoder_layers=6,
        decoder_heads=8,
        decoder_ffn_dim=2048,
        num_queries=100,
        num_labels=150,
        input_image_shape=512,
        name="MaskFormerModel",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1
        input_image_shape = standardize_input_shape(input_image_shape, data_format)

        pixel_values = layers.Input(shape=input_image_shape, name="pixel_values")

        outputs = maskformer_functional(
            pixel_values,
            backbone_embed_dim=backbone_embed_dim,
            backbone_depths=backbone_depths,
            backbone_num_heads=backbone_num_heads,
            backbone_window_size=backbone_window_size,
            fpn_feature_size=fpn_feature_size,
            mask_feature_size=mask_feature_size,
            decoder_d_model=decoder_d_model,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            num_queries=num_queries,
            num_labels=num_labels,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        super().__init__(inputs=pixel_values, outputs=outputs, name=name, **kwargs)

        self.backbone_embed_dim = backbone_embed_dim
        self.backbone_depths = tuple(backbone_depths)
        self.backbone_num_heads = tuple(backbone_num_heads)
        self.backbone_window_size = backbone_window_size
        self.fpn_feature_size = fpn_feature_size
        self.mask_feature_size = mask_feature_size
        self.decoder_d_model = decoder_d_model
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_queries = num_queries
        self.num_labels = num_labels
        self.input_image_shape = input_image_shape

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "backbone_embed_dim": self.backbone_embed_dim,
                "backbone_depths": self.backbone_depths,
                "backbone_num_heads": self.backbone_num_heads,
                "backbone_window_size": self.backbone_window_size,
                "fpn_feature_size": self.fpn_feature_size,
                "mask_feature_size": self.mask_feature_size,
                "decoder_d_model": self.decoder_d_model,
                "decoder_layers": self.decoder_layers,
                "decoder_heads": self.decoder_heads,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "num_queries": self.num_queries,
                "num_labels": self.num_labels,
                "input_image_shape": self.input_image_shape,
                "name": self.name,
            }
        )
        return c

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class MaskFormerUniversalSegment(BaseModel):
    """MaskFormer universal segmentation model.

    Composes :class:`MaskFormerModel` and exposes the prediction output
    dict with ``class_queries_logits`` and ``masks_queries_logits`` keys
    matching HF's ``MaskFormerForInstanceSegmentation`` output.

    Reference:
    - [Per-Pixel Classification is Not All You Need for Semantic
      Segmentation](https://arxiv.org/abs/2107.06278)
    """

    BASE_MODEL_CONFIG = MASKFORMER_CONFIG
    BASE_WEIGHT_CONFIG = MASKFORMER_WEIGHTS
    HF_MODEL_TYPE = "maskformer"

    def __init__(
        self,
        backbone_embed_dim=96,
        backbone_depths=(2, 2, 6, 2),
        backbone_num_heads=(3, 6, 12, 24),
        backbone_window_size=7,
        fpn_feature_size=256,
        mask_feature_size=256,
        decoder_d_model=256,
        decoder_layers=6,
        decoder_heads=8,
        decoder_ffn_dim=2048,
        num_queries=100,
        num_labels=150,
        input_image_shape=512,
        name="MaskFormerUniversalSegment",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1
        input_image_shape = standardize_input_shape(input_image_shape, data_format)

        pixel_values = layers.Input(shape=input_image_shape, name="pixel_values")

        outputs = maskformer_functional(
            pixel_values,
            backbone_embed_dim=backbone_embed_dim,
            backbone_depths=backbone_depths,
            backbone_num_heads=backbone_num_heads,
            backbone_window_size=backbone_window_size,
            fpn_feature_size=fpn_feature_size,
            mask_feature_size=mask_feature_size,
            decoder_d_model=decoder_d_model,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            num_queries=num_queries,
            num_labels=num_labels,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        super().__init__(inputs=pixel_values, outputs=outputs, name=name, **kwargs)

        self.backbone_embed_dim = backbone_embed_dim
        self.backbone_depths = tuple(backbone_depths)
        self.backbone_num_heads = tuple(backbone_num_heads)
        self.backbone_window_size = backbone_window_size
        self.fpn_feature_size = fpn_feature_size
        self.mask_feature_size = mask_feature_size
        self.decoder_d_model = decoder_d_model
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_queries = num_queries
        self.num_labels = num_labels
        self.input_image_shape = input_image_shape

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "backbone_embed_dim": self.backbone_embed_dim,
                "backbone_depths": self.backbone_depths,
                "backbone_num_heads": self.backbone_num_heads,
                "backbone_window_size": self.backbone_window_size,
                "fpn_feature_size": self.fpn_feature_size,
                "mask_feature_size": self.mask_feature_size,
                "decoder_d_model": self.decoder_d_model,
                "decoder_layers": self.decoder_layers,
                "decoder_heads": self.decoder_heads,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "num_queries": self.num_queries,
                "num_labels": self.num_labels,
                "input_image_shape": self.input_image_shape,
                "name": self.name,
            }
        )
        return c

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def config_from_hf(cls, hf_config):
        """Map a HuggingFace MaskFormer config to constructor kwargs.

        Reads the Swin backbone and DETR-style decoder sub-configs and returns
        the keyword arguments for building the equivalent Keras model.

        Args:
            hf_config: HuggingFace ``MaskFormerConfig`` as a dict.

        Returns:
            Dict of constructor keyword arguments for this model class.
        """
        backbone = hf_config.get("backbone_config", {})
        decoder = hf_config.get("decoder_config", {})
        depths = backbone.get("depths", [2, 2, 6, 2])
        num_heads = backbone.get("num_heads", [3, 6, 12, 24])

        from kerasformers.base.base_model import hf_num_labels

        return {
            "backbone_embed_dim": backbone.get("embed_dim", 96),
            "backbone_depths": tuple(depths),
            "backbone_num_heads": tuple(num_heads),
            "backbone_window_size": backbone.get("window_size", 7),
            "fpn_feature_size": hf_config.get("fpn_feature_size", 256),
            "mask_feature_size": hf_config.get("mask_feature_size", 256),
            "decoder_d_model": decoder.get("d_model", 256),
            "decoder_layers": decoder.get("decoder_layers", 6),
            "decoder_heads": decoder.get("decoder_attention_heads", 8),
            "decoder_ffn_dim": decoder.get("decoder_ffn_dim", 2048),
            "num_queries": decoder.get("num_queries", 100),
            "num_labels": hf_num_labels(hf_config),
            "input_image_shape": backbone.get("image_size", 384),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        """Copy weights from a HuggingFace MaskFormer checkpoint into the model.

        Args:
            keras_model: The freshly-built Keras model to populate.
            hf_state_dict: The HuggingFace model ``state_dict`` (numpy arrays).
        """
        from .convert_maskformer_hf_to_keras import transfer_maskformer_weights

        transfer_maskformer_weights(keras_model, hf_state_dict)
