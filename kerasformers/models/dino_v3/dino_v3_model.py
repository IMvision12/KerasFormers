import keras
from keras import layers, ops, utils

from kerasformers.base import BaseModel
from kerasformers.layers import ImageNormalizationLayer, LayerScale
from kerasformers.models.convnext.convnext_model import convnext_backbone_feature
from kerasformers.models.dino_v3.convert_dino_v3_hf_to_keras import (
    transfer_dinov3_convnext_weights,
    transfer_dinov3_vit_weights,
)

from .config import (
    DINOV3_CONVNEXT_CONFIG,
    DINOV3_CONVNEXT_WEIGHTS,
    DINOV3_VIT_CONFIG,
    DINOV3_VIT_WEIGHTS,
)
from .dino_v3_layers import (
    DinoV3Attention,
    DinoV3CLSToken,
    DinoV3RegisterTokens,
    build_rope_2d_cache,
)


def dinov3_swiglu_ffn(x, dim, hidden_dim, block_idx, hidden_act="gelu", mlp_bias=True):
    """Gated feed-forward network used in DINOv3 ViT blocks (GeGLU / SwiGLU)."""
    gate = layers.Dense(
        hidden_dim, use_bias=mlp_bias, name=f"blocks_{block_idx}_swiglu_gate"
    )(x)
    gate = layers.Activation(hidden_act)(gate)
    up = layers.Dense(
        hidden_dim, use_bias=mlp_bias, name=f"blocks_{block_idx}_swiglu_up"
    )(x)
    x = layers.Multiply()([gate, up])
    x = layers.Dense(dim, use_bias=mlp_bias, name=f"blocks_{block_idx}_swiglu_down")(x)
    return x


def dinov3_mlp_block(x, dim, hidden_dim, block_idx, hidden_act="gelu", mlp_bias=True):
    """Standard two-layer MLP with configurable activation."""
    x = layers.Dense(hidden_dim, use_bias=mlp_bias, name=f"blocks_{block_idx}_dense_1")(
        x
    )
    x = layers.Activation(hidden_act, name=f"blocks_{block_idx}_{hidden_act}")(x)
    x = layers.Dense(dim, use_bias=mlp_bias, name=f"blocks_{block_idx}_dense_2")(x)
    return x


def dinov3_transformer_block(
    inputs,
    dim,
    num_heads,
    mlp_hidden_dim,
    num_prefix_tokens,
    rope_theta,
    use_swiglu,
    init_values,
    block_idx,
    rope_cos,
    rope_sin,
    query_bias=True,
    key_bias=False,
    value_bias=True,
    hidden_act="gelu",
    mlp_bias=True,
    layer_norm_eps=1e-5,
):
    """DINOv3 transformer block with 2D-RoPE self-attention + MLP."""
    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, axis=-1, name=f"blocks_{block_idx}_layernorm_1"
    )(inputs)
    attn = DinoV3Attention(
        dim=dim,
        num_heads=num_heads,
        num_prefix_tokens=num_prefix_tokens,
        rope_theta=rope_theta,
        query_bias=query_bias,
        key_bias=key_bias,
        value_bias=value_bias,
        block_prefix=f"blocks_{block_idx}",
    )
    attn.set_rope_cache(rope_cos, rope_sin)
    x = attn(x)
    if init_values is not None:
        x = LayerScale(
            init_values=init_values, name=f"blocks_{block_idx}_layerscale_1"
        )(x)
    x = layers.Add(name=f"blocks_{block_idx}_add_1")([x, inputs])

    y = layers.LayerNormalization(
        epsilon=layer_norm_eps, axis=-1, name=f"blocks_{block_idx}_layernorm_2"
    )(x)
    if use_swiglu:
        y = dinov3_swiglu_ffn(y, dim, mlp_hidden_dim, block_idx, hidden_act, mlp_bias)
    else:
        y = dinov3_mlp_block(y, dim, mlp_hidden_dim, block_idx, hidden_act, mlp_bias)
    if init_values is not None:
        y = LayerScale(
            init_values=init_values, name=f"blocks_{block_idx}_layerscale_2"
        )(y)
    out = layers.Add(name=f"blocks_{block_idx}_add_2")([x, y])
    return out


@keras.saving.register_keras_serializable(package="kerasformers")
class DinoV3ViTBackbone(BaseModel):
    """DINOv3 Vision Transformer backbone with 2D RoPE and register tokens.

    Returns the list of intermediate feature maps (initial embedding +
    each transformer block), suitable for feeding into detection /
    segmentation / depth necks.

    Weights are gated on HuggingFace — see
    https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m for
    license acceptance. Accept the license and set ``HF_TOKEN`` env var
    before calling :meth:`from_weights`.

    Reference:
        - `DINOv3 <https://arxiv.org/abs/2508.10104>`_

    Args:
        patch_size: ViT patch size. DINOv3 uses 16.
        dim: Hidden dimension.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        mlp_ratio: MLP expansion ratio. Defaults to ``4.0``.
        use_swiglu: Whether to use a gated MLP (GeGLU / SwiGLU)
            instead of the standard two-layer MLP. Defaults to ``False``.
        num_register_tokens: Number of register tokens. Defaults to ``4``.
        init_values: LayerScale init value. Defaults to ``1.0``.
        rope_theta: 2D-RoPE frequency base. Defaults to ``100.0``.
        query_bias: Whether the attention Q projection uses bias.
            Defaults to ``True`` (canonical DINOv3 setting).
        key_bias: Whether the attention K projection uses bias.
            Defaults to ``False`` (canonical DINOv3 setting).
        value_bias: Whether the attention V projection uses bias.
            Defaults to ``True`` (canonical DINOv3 setting).
        hidden_act: MLP activation name (``"gelu"`` or ``"silu"``).
            Defaults to ``"gelu"``.
        mlp_bias: Whether MLP Dense layers use bias. Defaults to ``True``.
        layer_norm_eps: Epsilon for LayerNorm layers. Defaults to ``1e-5``.
        include_normalization: Whether to prepend
            :class:`ImageNormalizationLayer`.
        normalization_mode: Normalization preset.
        input_shape: Image input shape excluding batch dim. Defaults to
            ``(224, 224, 3)``.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = DINOV3_VIT_CONFIG
    BASE_WEIGHT_CONFIG = DINOV3_VIT_WEIGHTS
    HF_MODEL_TYPE = "dinov3_vit"

    @classmethod
    def config_from_hf(cls, hf_config):
        intermediate = hf_config.get("intermediate_size")
        hidden = hf_config["hidden_size"]
        mlp_ratio = intermediate / hidden if intermediate else 4.0
        return {
            "patch_size": hf_config.get("patch_size", 16),
            "dim": hidden,
            "depth": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "mlp_ratio": mlp_ratio,
            "use_swiglu": hf_config.get("use_gated_mlp", False),
            "num_register_tokens": hf_config.get("num_register_tokens", 4),
            "init_values": hf_config.get("layerscale_value", 1.0),
            "rope_theta": hf_config.get("rope_theta", 100.0),
            "query_bias": hf_config.get("query_bias", True),
            "key_bias": hf_config.get("key_bias", False),
            "value_bias": hf_config.get("value_bias", True),
            "hidden_act": hf_config.get("hidden_act", "gelu"),
            "mlp_bias": hf_config.get("mlp_bias", True),
            "layer_norm_eps": hf_config.get("layer_norm_eps", 1e-5),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        transfer_dinov3_vit_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        patch_size=16,
        dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        use_swiglu=False,
        num_register_tokens=4,
        init_values=1.0,
        rope_theta=100.0,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        hidden_act="gelu",
        mlp_bias=True,
        layer_norm_eps=1e-5,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="DinoV3ViTBackbone",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        if input_shape is None and input_tensor is None:
            input_shape = (224, 224, 3)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        if data_format == "channels_first":
            height, width = input_shape[1], input_shape[2]
        else:
            height, width = input_shape[0], input_shape[1]

        grid_h = height // patch_size
        grid_w = width // patch_size
        num_prefix_tokens = 1 + num_register_tokens

        rope_cos_np, rope_sin_np = build_rope_2d_cache(
            grid_h, grid_w, dim // num_heads, theta=rope_theta
        )
        rope_cos = ops.convert_to_tensor(rope_cos_np)
        rope_sin = ops.convert_to_tensor(rope_sin_np)

        mlp_hidden_dim = int(dim * mlp_ratio)

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(img_input)
            if include_normalization
            else img_input
        )
        x = layers.Conv2D(
            filters=dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format=data_format,
            name="patch_embed",
        )(x)
        x = layers.Reshape((-1, dim))(x)
        x = DinoV3CLSToken(name="cls_token")(x)
        if num_register_tokens > 0:
            x = DinoV3RegisterTokens(
                num_tokens=num_register_tokens, name="register_tokens"
            )(x)

        features = [x]
        for i in range(depth):
            x = dinov3_transformer_block(
                x,
                dim=dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                num_prefix_tokens=num_prefix_tokens,
                rope_theta=rope_theta,
                use_swiglu=use_swiglu,
                init_values=init_values,
                block_idx=i,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                query_bias=query_bias,
                key_bias=key_bias,
                value_bias=value_bias,
                hidden_act=hidden_act,
                mlp_bias=mlp_bias,
                layer_norm_eps=layer_norm_eps,
            )
            features.append(x)
        # Final LayerNorm applied to last feature
        features[-1] = layers.LayerNormalization(
            epsilon=layer_norm_eps, axis=-1, name="final_layernorm"
        )(features[-1])

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_swiglu = use_swiglu
        self.num_register_tokens = num_register_tokens
        self.init_values = init_values
        self.rope_theta = rope_theta
        self.query_bias = query_bias
        self.key_bias = key_bias
        self.value_bias = value_bias
        self.hidden_act = hidden_act
        self.mlp_bias = mlp_bias
        self.layer_norm_eps = layer_norm_eps
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "dim": self.dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "use_swiglu": self.use_swiglu,
                "num_register_tokens": self.num_register_tokens,
                "init_values": self.init_values,
                "rope_theta": self.rope_theta,
                "query_bias": self.query_bias,
                "key_bias": self.key_bias,
                "value_bias": self.value_bias,
                "hidden_act": self.hidden_act,
                "mlp_bias": self.mlp_bias,
                "layer_norm_eps": self.layer_norm_eps,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self._input_shape_val,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class DinoV3ConvNeXtBackbone(BaseModel):
    """DINOv3 ConvNeXt backbone.

    Returns the list of intermediate feature maps from each ConvNeXt
    stage, suitable for feeding into detection / segmentation / depth
    necks.

    Weights are gated on HuggingFace — see
    https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
    for license acceptance. Accept the license and set ``HF_TOKEN`` env
    var before calling :meth:`from_weights`.

    Reference:
        - `DINOv3 <https://arxiv.org/abs/2508.10104>`_

    Args:
        depths: Per-stage block counts.
        projection_dims: Per-stage channel counts.
        include_normalization: Whether to prepend
            :class:`ImageNormalizationLayer`.
        normalization_mode: Normalization preset.
        input_shape: Image input shape excluding batch dim. Defaults to
            ``(224, 224, 3)``.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = DINOV3_CONVNEXT_CONFIG
    BASE_WEIGHT_CONFIG = DINOV3_CONVNEXT_WEIGHTS
    HF_MODEL_TYPE = "dinov3_convnext"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "depths": list(hf_config["depths"]),
            "projection_dims": list(hf_config["hidden_sizes"]),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        transfer_dinov3_convnext_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        depths=None,
        projection_dims=None,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="DinoV3ConvNeXtBackbone",
        **kwargs,
    ):
        if depths is None:
            depths = [3, 3, 9, 3]
        if projection_dims is None:
            projection_dims = [96, 192, 384, 768]
        if input_shape is None and input_tensor is None:
            input_shape = (224, 224, 3)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(img_input)
            if include_normalization
            else img_input
        )
        features = convnext_backbone_feature(
            x,
            depths=depths,
            projection_dims=projection_dims,
            drop_path_rate=0.0,
            layer_scale_init_value=1e-6,
            use_conv=True,
            use_grn=False,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=True,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.depths = list(depths)
        self.projection_dims = list(projection_dims)
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "projection_dims": self.projection_dims,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self._input_shape_val,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
