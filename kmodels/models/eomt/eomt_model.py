import keras
from keras import layers, ops, utils

from kmodels.base import BaseModel
from kmodels.base.base_model import hf_num_labels
from kmodels.layers import StochasticDepth

from .config import EOMT_CONFIG, EOMT_WEIGHTS
from .eomt_layers import (
    EoMTAttention,
    EoMTEmbeddings,
    EoMTLayerScale,
    EoMTQueryInjection,
)


def eomt_mlp(x, hidden_size, mlp_ratio=4, block_prefix="layers_0"):
    """Standard two-layer MLP with GELU activation."""
    hidden_features = int(hidden_size * mlp_ratio)
    x = layers.Dense(hidden_features, name=f"{block_prefix}_mlp_fc1")(x)
    x = layers.Activation("gelu", name=f"{block_prefix}_mlp_gelu")(x)
    x = layers.Dense(hidden_size, name=f"{block_prefix}_mlp_fc2")(x)
    return x


def eomt_swiglu_ffn(x, hidden_size, mlp_ratio=4, block_prefix="layers_0"):
    """SwiGLU gated feed-forward network."""
    hidden_features = int(hidden_size * mlp_ratio)
    hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    x = layers.Dense(2 * hidden_features, name=f"{block_prefix}_mlp_weights_in")(x)
    x1 = x[..., :hidden_features]
    x2 = x[..., hidden_features:]
    hidden = layers.Activation("silu", name=f"{block_prefix}_mlp_silu")(x1)
    hidden = layers.Multiply(name=f"{block_prefix}_mlp_gate")([hidden, x2])
    return layers.Dense(hidden_size, name=f"{block_prefix}_mlp_weights_out")(hidden)


def eomt_encoder_layer(
    hidden_states,
    hidden_size,
    num_heads,
    mlp_ratio=4,
    layerscale_value=1.0,
    drop_path_rate=0.0,
    attention_dropout=0.0,
    use_swiglu_ffn=False,
    layer_norm_eps=1e-6,
    block_prefix="layers_0",
):
    """Single EoMT transformer encoder layer with pre-norm design."""
    residual = hidden_states
    hidden_states = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_norm1"
    )(hidden_states)
    hidden_states = EoMTAttention(
        hidden_size, num_heads, attention_dropout, name=f"{block_prefix}_attention"
    )(hidden_states)
    hidden_states = EoMTLayerScale(
        init_value=layerscale_value, name=f"{block_prefix}_layer_scale1"
    )(hidden_states)
    drop_path = (
        StochasticDepth(drop_path_rate, name=f"{block_prefix}_drop_path")
        if drop_path_rate > 0.0
        else layers.Identity(name=f"{block_prefix}_identity")
    )
    hidden_states = layers.Add(name=f"{block_prefix}_attn_residual")(
        [drop_path(hidden_states), residual]
    )

    residual = hidden_states
    hidden_states = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_norm2"
    )(hidden_states)

    if use_swiglu_ffn:
        hidden_states = eomt_swiglu_ffn(
            hidden_states, hidden_size, mlp_ratio, block_prefix
        )
    else:
        hidden_states = eomt_mlp(hidden_states, hidden_size, mlp_ratio, block_prefix)

    hidden_states = EoMTLayerScale(
        init_value=layerscale_value, name=f"{block_prefix}_layer_scale2"
    )(hidden_states)
    hidden_states = layers.Add(name=f"{block_prefix}_mlp_residual")(
        [drop_path(hidden_states), residual]
    )
    return hidden_states


def eomt_scale_layer(x, hidden_size, data_format, block_prefix="upscale_block_0"):
    """Single 2x spatial upscaling layer for mask feature decoding."""
    x = layers.Conv2DTranspose(
        hidden_size,
        kernel_size=2,
        strides=2,
        padding="valid",
        use_bias=True,
        data_format=data_format,
        name=f"{block_prefix}_conv1",
    )(x)
    x = layers.Activation("gelu", name=f"{block_prefix}_gelu")(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{block_prefix}_conv2",
    )(x)
    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1))(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm")(x)
    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2))(x)
    return x


def eomt_scale_block(x, hidden_size, num_upscale_blocks, data_format):
    """Stack of spatial upscaling layers."""
    for i in range(num_upscale_blocks):
        x = eomt_scale_layer(
            x, hidden_size, data_format=data_format, block_prefix=f"upscale_block_{i}"
        )
    return x


def eomt_mask_head(x, hidden_size):
    """Mask prediction head with three dense layers and GELU activations."""
    x = layers.Dense(hidden_size, name="mask_head_fc1")(x)
    x = layers.Activation("gelu", name="mask_head_gelu1")(x)
    x = layers.Dense(hidden_size, name="mask_head_fc2")(x)
    x = layers.Activation("gelu", name="mask_head_gelu2")(x)
    x = layers.Dense(hidden_size, name="mask_head_fc3")(x)
    return x


def eomt_functional(
    inputs,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    num_blocks,
    num_queries,
    layerscale_value,
    patch_size,
    num_register_tokens,
    mlp_ratio,
    drop_path_rate,
    attention_dropout,
    use_swiglu_ffn,
    layer_norm_eps,
):
    """Build the EoMT encoder graph (no task heads).

    Patch + register + CLS embeddings → ``num_hidden_layers`` encoder
    blocks with the final ``num_blocks`` receiving injected object
    queries → final ``LayerNormalization``. Returns the full sequence
    output.

    Reference:
        - `Your ViT is Secretly an Image Segmentation Model
          <https://arxiv.org/abs/2503.19108>`_
    """
    data_format = keras.config.image_data_format()
    image_size = inputs.shape[2] if data_format == "channels_first" else inputs.shape[1]

    hidden_states = EoMTEmbeddings(
        hidden_size=hidden_size,
        patch_size=patch_size,
        image_size=image_size,
        num_register_tokens=num_register_tokens,
        name="embeddings",
    )(inputs)

    query_injection = EoMTQueryInjection(num_queries, hidden_size, name="query")
    query_injection_idx = num_hidden_layers - num_blocks

    for i in range(num_hidden_layers):
        if i == query_injection_idx:
            hidden_states = query_injection(hidden_states)

        hidden_states = eomt_encoder_layer(
            hidden_states,
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            layerscale_value=layerscale_value,
            drop_path_rate=drop_path_rate,
            attention_dropout=attention_dropout,
            use_swiglu_ffn=use_swiglu_ffn,
            layer_norm_eps=layer_norm_eps,
            block_prefix=f"layers_{i}",
        )

    sequence_output = layers.LayerNormalization(
        epsilon=layer_norm_eps, name="layernorm"
    )(hidden_states)
    return sequence_output


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTModel(BaseModel):
    """EoMT encoder backbone with query injection (no task heads).

    Builds the plain DINOv2-style ViT encoder used by EoMT, including
    the learned object-query injection at the boundary of the final
    ``num_blocks`` encoder layers. Returns the post-LayerNorm sequence
    output of shape
    ``(batch, num_queries + num_prefix + num_patches, hidden_size)``.
    Pair with :class:`EoMTSegment` to get the full universal-
    segmentation outputs (class logits + mask logits).

    Reference:
        - `Your ViT is Secretly an Image Segmentation Model
          <https://arxiv.org/abs/2503.19108>`_

    Args:
        hidden_size: Transformer hidden dimension.
        num_hidden_layers: Total number of transformer encoder layers.
        num_attention_heads: Number of attention heads per layer.
        num_blocks: Number of final encoder blocks that receive the
            injected object queries.
        num_queries: Number of learned object queries.
        layerscale_value: Initial value for LayerScale parameters.
        patch_size: Image patch size.
        num_register_tokens: Number of DINOv2-style register tokens.
        mlp_ratio: Expansion ratio for the feedforward network.
        drop_path_rate: Stochastic depth rate.
        attention_dropout: Dropout rate for the attention weights.
        use_swiglu_ffn: Whether to use SwiGLU instead of the GELU MLP.
        layer_norm_eps: Epsilon for layer normalization.
        input_shape: Image input shape excluding batch dim.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    KMODELS_CONFIG = EOMT_CONFIG
    KMODELS_WEIGHTS = None
    HF_MODEL_TYPE = "eomt"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_blocks=4,
        num_queries=200,
        layerscale_value=1e-5,
        patch_size=16,
        num_register_tokens=4,
        mlp_ratio=4,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        use_swiglu_ffn=False,
        layer_norm_eps=1e-6,
        input_shape=None,
        input_tensor=None,
        name="EoMTModel",
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (640, 640, 3)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        sequence_output = eomt_functional(
            img_input,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_blocks=num_blocks,
            num_queries=num_queries,
            layerscale_value=layerscale_value,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            attention_dropout=attention_dropout,
            use_swiglu_ffn=use_swiglu_ffn,
            layer_norm_eps=layer_norm_eps,
        )

        super().__init__(inputs=img_input, outputs=sequence_output, name=name, **kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_blocks = num_blocks
        self.num_queries = num_queries
        self.layerscale_value = layerscale_value
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.use_swiglu_ffn = use_swiglu_ffn
        self.layer_norm_eps = layer_norm_eps
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_blocks": self.num_blocks,
                "num_queries": self.num_queries,
                "layerscale_value": self.layerscale_value,
                "patch_size": self.patch_size,
                "num_register_tokens": self.num_register_tokens,
                "mlp_ratio": self.mlp_ratio,
                "drop_path_rate": self.drop_path_rate,
                "attention_dropout": self.attention_dropout,
                "use_swiglu_ffn": self.use_swiglu_ffn,
                "layer_norm_eps": self.layer_norm_eps,
                "input_shape": self._input_shape_val,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class EoMTSegment(BaseModel):
    """EoMT full universal-segmentation model (encoder + class + mask heads).

    Composes :class:`EoMTModel` and adds the class-prediction head, the
    mask-feature upscale stack, the per-query mask head, and the
    bilinear einsum that produces per-query mask logits. Output is a
    dict with:

    - ``"class_logits"``: ``(batch, num_queries, num_labels + 1)``
    - ``"mask_logits"``: ``(batch, num_queries, H_up, W_up)`` where
      ``H_up = image_size // patch_size * 2^num_upscale_blocks``.

    Reference:
        - `Your ViT is Secretly an Image Segmentation Model
          <https://arxiv.org/abs/2503.19108>`_

    Args:
        num_labels: Number of segmentation classes.
        num_upscale_blocks: Number of 2x upscaling layers applied to
            patch features before mask prediction.
        See :class:`EoMTModel` for the remaining args.
    """

    KMODELS_CONFIG = EOMT_CONFIG
    KMODELS_WEIGHTS = EOMT_WEIGHTS
    HF_MODEL_TYPE = "eomt"

    @classmethod
    def config_from_hf(cls, hf_config):
        image_size = hf_config.get("image_size", 640)
        return {
            "hidden_size": hf_config["hidden_size"],
            "num_hidden_layers": hf_config["num_hidden_layers"],
            "num_attention_heads": hf_config["num_attention_heads"],
            "num_blocks": hf_config["num_blocks"],
            "num_queries": hf_config["num_queries"],
            "layerscale_value": hf_config.get("layerscale_value", 1.0),
            "patch_size": hf_config.get("patch_size", 16),
            "num_register_tokens": hf_config.get("num_register_tokens", 4),
            "mlp_ratio": hf_config.get("mlp_ratio", 4),
            "use_swiglu_ffn": hf_config.get("use_swiglu_ffn", False),
            "num_upscale_blocks": hf_config.get("num_upscale_blocks", 2),
            "num_labels": hf_num_labels(hf_config),
            "input_shape": (image_size, image_size, 3),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kmodels.models.eomt.convert_eomt_hf_to_keras import (
            transfer_eomt_weights,
        )

        transfer_eomt_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_blocks=4,
        num_queries=200,
        num_labels=133,
        layerscale_value=1e-5,
        patch_size=16,
        num_register_tokens=4,
        num_upscale_blocks=2,
        mlp_ratio=4,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        use_swiglu_ffn=False,
        layer_norm_eps=1e-6,
        input_shape=None,
        input_tensor=None,
        name="EoMTSegment",
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (640, 640, 3)

        base = EoMTModel(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_blocks=num_blocks,
            num_queries=num_queries,
            layerscale_value=layerscale_value,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            attention_dropout=attention_dropout,
            use_swiglu_ffn=use_swiglu_ffn,
            layer_norm_eps=layer_norm_eps,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_model",
        )
        sequence_output = base.output

        data_format = keras.config.image_data_format()
        image_size = (
            input_shape[1] if data_format == "channels_first" else input_shape[0]
        )
        grid_h = grid_w = image_size // patch_size
        num_prefix_tokens = 1 + num_register_tokens

        query_output = sequence_output[:, :num_queries, :]
        patch_output = sequence_output[:, num_queries + num_prefix_tokens :, :]

        class_logits = layers.Dense(num_labels + 1, name="class_predictor")(
            query_output
        )

        query_mask_tokens = eomt_mask_head(query_output, hidden_size)

        if data_format == "channels_first":
            patch_spatial = ops.reshape(patch_output, (-1, hidden_size, grid_h, grid_w))
        else:
            patch_spatial = ops.reshape(patch_output, (-1, grid_h, grid_w, hidden_size))

        upscaled_features = eomt_scale_block(
            patch_spatial, hidden_size, num_upscale_blocks, data_format=data_format
        )

        if data_format == "channels_first":
            mask_logits = ops.einsum(
                "bqc,bchw->bqhw", query_mask_tokens, upscaled_features
            )
        else:
            mask_logits = ops.einsum(
                "bqc,bhwc->bqhw", query_mask_tokens, upscaled_features
            )

        super().__init__(
            inputs=base.input,
            outputs={"class_logits": class_logits, "mask_logits": mask_logits},
            name=name,
            **kwargs,
        )

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_blocks = num_blocks
        self.num_queries = num_queries
        self.num_labels = num_labels
        self.layerscale_value = layerscale_value
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.num_upscale_blocks = num_upscale_blocks
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.use_swiglu_ffn = use_swiglu_ffn
        self.layer_norm_eps = layer_norm_eps
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_blocks": self.num_blocks,
                "num_queries": self.num_queries,
                "num_labels": self.num_labels,
                "layerscale_value": self.layerscale_value,
                "patch_size": self.patch_size,
                "num_register_tokens": self.num_register_tokens,
                "num_upscale_blocks": self.num_upscale_blocks,
                "mlp_ratio": self.mlp_ratio,
                "drop_path_rate": self.drop_path_rate,
                "attention_dropout": self.attention_dropout,
                "use_swiglu_ffn": self.use_swiglu_ffn,
                "layer_norm_eps": self.layer_norm_eps,
                "input_shape": self._input_shape_val,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
