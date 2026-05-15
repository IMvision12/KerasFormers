import keras
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer, LayerScale, StochasticDepth
from kmodels.models.cait.cait_layers import (
    AddPositionEmbs,
    ClassAttention,
    ClassDistToken,
    TalkingHeadAttention,
)
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import CAIT_MODEL_CONFIG, CAIT_WEIGHT_CONFIG
from .convert_cait_torch_to_keras import transfer_cait_weights


def mlp_block(x, hidden_dim, out_dim, drop_rate=0.0, block_prefix=None):
    """Two-layer MLP block: Dense -> GELU -> Drop -> Dense -> Drop.

    Args:
        x: Input token tensor of shape ``(B, N, D)``.
        hidden_dim: Output dimension of the first Dense layer.
        out_dim: Output dimension of the second Dense layer.
        drop_rate: Dropout rate applied after each Dense.
        block_prefix: Optional prefix used to name the inner Dense layers.

    Returns:
        Tensor of shape ``(B, N, out_dim)``.
    """
    x = layers.Dense(
        hidden_dim,
        activation="gelu",
        name=f"{block_prefix}_dense_1" if block_prefix else None,
    )(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(out_dim, name=f"{block_prefix}_dense_2" if block_prefix else None)(
        x
    )
    x = layers.Dropout(drop_rate)(x)
    return x


def layer_scale_talking_head_block(
    x,
    embed_dim,
    num_heads,
    mlp_ratio=4.0,
    drop_rate=0.0,
    init_values=1e-5,
    block_prefix="block",
):
    """CaiT main block: LN -> TalkingHeadAttn -> LayerScale -> SD -> Add -> LN -> MLP -> LayerScale -> SD -> Add.

    Args:
        x: Input token tensor of shape ``(B, N, embed_dim)``.
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden expansion ratio for the MLP block.
        drop_rate: Stochastic-depth drop rate applied to each residual branch.
        init_values: Initial value for the LayerScale per-channel gamma.
        block_prefix: Prefix used to name layers inside the block.

    Returns:
        Tensor of shape ``(B, N, embed_dim)`` after both residual branches.
    """
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_1")(x)
    attn = TalkingHeadAttention(
        dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        block_prefix=f"{block_prefix}_attn",
    )(y)
    attn = LayerScale(init_values=init_values, name=f"{block_prefix}_layerscale_1")(
        attn
    )
    if drop_rate > 0:
        attn = StochasticDepth(drop_rate)(attn)
    x = layers.Add(name=f"{block_prefix}_add_1")([x, attn])

    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_2")(x)
    mlp = mlp_block(
        y,
        hidden_dim=int(embed_dim * mlp_ratio),
        out_dim=embed_dim,
        block_prefix=f"{block_prefix}_mlp",
    )
    mlp = LayerScale(init_values=init_values, name=f"{block_prefix}_layerscale_2")(mlp)
    if drop_rate > 0:
        mlp = StochasticDepth(drop_rate)(mlp)
    return layers.Add(name=f"{block_prefix}_add_2")([x, mlp])


def layer_scale_class_attn_block(
    cls_token,
    x,
    embed_dim,
    num_heads,
    mlp_ratio=4.0,
    init_values=1e-5,
    block_prefix="block_token_only",
):
    """Class-attention-only block: cls_token attends to patch tokens, then MLP.

    Args:
        cls_token: Class token tensor of shape ``(B, 1, embed_dim)``.
        x: Patch tokens of shape ``(B, N, embed_dim)``.
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden expansion ratio for the MLP block.
        init_values: Initial value for the LayerScale per-channel gamma.
        block_prefix: Prefix used to name layers inside the block.

    Returns:
        Updated ``cls_token`` tensor of shape ``(B, 1, embed_dim)``.
    """
    concat = layers.Concatenate(axis=1)([cls_token, x])
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_1")(
        concat
    )
    cls = ClassAttention(
        dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        block_prefix=f"{block_prefix}_attn",
    )(y)
    cls = LayerScale(init_values=init_values, name=f"{block_prefix}_layerscale_1")(cls)
    cls_token = layers.Add(name=f"{block_prefix}_add_1")([cls_token, cls])

    y = layers.LayerNormalization(epsilon=1e-6, name=f"{block_prefix}_layernorm_2")(
        cls_token
    )
    mlp = mlp_block(
        y,
        hidden_dim=int(embed_dim * mlp_ratio),
        out_dim=embed_dim,
        block_prefix=f"{block_prefix}_mlp",
    )
    mlp = LayerScale(init_values=init_values, name=f"{block_prefix}_layerscale_2")(mlp)
    return layers.Add(name=f"{block_prefix}_add_2")([cls_token, mlp])


def cait_backbone_feature(
    inputs,
    *,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    drop_path_rate,
    image_size,
    data_format,
    depth_token_only=2,
    return_stages=False,
):
    """CaiT stem + talking-head blocks + class-attn blocks.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        patch_size: Conv-stem patch size in pixels.
        embed_dim: Token embedding dimension.
        depth: Number of TalkingHead transformer blocks.
        num_heads: Number of attention heads per block.
        drop_path_rate: Maximum stochastic-depth drop rate (linearly scaled
            across the ``depth`` blocks).
        image_size: Input image resolution (documentation only).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        depth_token_only: Number of trailing class-attention blocks.
        return_stages: If ``True``, return a list of per-block (talking-head
            + class-attn) intermediate outputs ending with the final-LN
            output. Otherwise return only the final-LN output.

    Returns:
        ``(B, 1+N, D)`` tensor of final-LN normalized tokens — CLS at index 0
        followed by ``N = (H/patch_size) * (W/patch_size)`` patch tokens.
        When ``return_stages=True``, returns a list of intermediate tensors;
        the last entry is the same final-LN output.
    """
    x = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        data_format=data_format,
        name="stem_conv",
    )(inputs)

    if data_format == "channels_first":
        grid_h = inputs.shape[2] // patch_size
        grid_w = inputs.shape[3] // patch_size
    else:
        grid_h = inputs.shape[1] // patch_size
        grid_w = inputs.shape[2] // patch_size

    x = layers.Reshape((-1, embed_dim))(x)
    x = AddPositionEmbs(
        grid_h=grid_h, grid_w=grid_w, no_embed_class=True, name="pos_embed"
    )(x)

    stages = []
    dpr = list(ops.linspace(0.0, drop_path_rate, depth))
    for i in range(depth):
        x = layer_scale_talking_head_block(
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=dpr[i],
            init_values=1e-5,
            block_prefix=f"blocks_{i}",
        )
        stages.append(x)

    cls_token = ClassDistToken(name="cls_token")(x)
    for i in range(depth_token_only):
        cls_token = layer_scale_class_attn_block(
            cls_token,
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            init_values=1e-5,
            block_prefix=f"blocks_token_only_{i}",
        )
        stages.append(cls_token)

    x = layers.Concatenate(axis=1, name="cat_cls_patch")([cls_token, x])
    x = layers.LayerNormalization(epsilon=1e-6, name="final_layernorm")(x)
    stages.append(x)
    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class CaiTModel(BaseModel):
    """CaiT backbone — the main feature extractor.

    Returns the final-LN normalized token sequence ``(B, 1+N, D)`` — CLS
    at index 0, then ``N = (H/patch_size) * (W/patch_size)`` patch tokens.
    This is the last layer output before the classifier head.
    :class:`CaiTClassify` composes this model and reads ``[:, 0]`` from
    the output to produce logits.

    Reference:
        Touvron et al., *Going deeper with Image Transformers*
        (https://arxiv.org/abs/2103.17239).

    Construction:

    >>> CaiTModel.from_weights("cait_s24_224_fb_dist_in1k")
    >>> CaiTModel.from_weights("timm:timm/cait_s24_224.fb_dist_in1k")
    """

    KMODELS_CONFIG = CAIT_MODEL_CONFIG
    KMODELS_WEIGHTS = CAIT_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = CaiTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_cait_weights(keras_model, state_dict)

    def __init__(
        self,
        as_backbone=False,
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=4,
        drop_path_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="CaiTModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=image_size,
            min_size=32,
            data_format=data_format,
            require_flatten=True,
            weights=None,
        )

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
        x = cait_backbone_feature(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            data_format=data_format,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.as_backbone = as_backbone
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "as_backbone": self.as_backbone,
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
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


@keras.saving.register_keras_serializable(package="kmodels")
class CaiTClassify(BaseModel):
    """CaiT image classifier — :class:`CaiTModel` + linear head on the CLS token.

    Wraps a :class:`CaiTModel` backbone and attaches a single Dense layer
    on the CLS token (index 0 of the backbone's output) to produce class
    logits. All architectural parameters are forwarded to the underlying
    :class:`CaiTModel`; only ``num_classes`` and ``classifier_activation``
    are head-specific.

    Reference:
        Touvron et al., *Going deeper with Image Transformers*
        (https://arxiv.org/abs/2103.17239).

    Args:
        patch_size: Conv-stem patch size in pixels.
        embed_dim: Token embedding dimension (192/288/384/768 for
            XXS/XS/S/M variants).
        depth: Number of patch-only talking-head blocks in the backbone.
        num_heads: Number of attention heads per block.
        drop_path_rate: Maximum stochastic-depth drop rate, linearly scaled
            across the ``depth`` patch blocks.
        image_size: Square input resolution.
        include_normalization: If True, the backbone prepends an
            :class:`~kmodels.layers.ImageNormalizationLayer` so inputs can
            be raw pixels in ``[0, 255]``.
        normalization_mode: Mode passed to the image normalization layer.
        input_shape: Optional explicit input shape ``(H, W, C)`` /
            ``(C, H, W)``. If None, derived from ``image_size``.
        input_tensor: Optional pre-built Keras tensor to use as input.
        num_classes: Output logits dimension.
        classifier_activation: Activation on the final Dense (``"linear"``
            returns raw logits; ``"softmax"`` returns probabilities).
        name: Model name. The internal backbone is named ``f"{name}_backbone"``.
        **kwargs: Forwarded to :class:`~kmodels.base.BaseModel`. ``timm_id``
            is consumed and dropped.

    Construction:

    >>> CaiTClassify.from_weights("cait_s24_224_fb_dist_in1k")
    >>> CaiTClassify.from_weights("timm:timm/cait_s24_224.fb_dist_in1k")
    """

    KMODELS_CONFIG = CAIT_MODEL_CONFIG
    KMODELS_WEIGHTS = CAIT_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_cait_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=4,
        drop_path_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="CaiTClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        backbone = CaiTModel(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(backbone.output[:, 0])

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
