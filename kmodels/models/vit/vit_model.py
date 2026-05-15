import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer, LayerScale
from kmodels.models.vit.vit_layers import (
    AddPositionEmbs,
    ClassDistToken,
    MultiHeadSelfAttention,
)
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import VIT_CONFIG, VIT_WEIGHTS
from .convert_vit_torch_to_keras import transfer_vit_weights


def mlp_block(inputs, hidden_features, out_features=None, drop=0.0, block_idx=0):
    """Standard transformer MLP block: Dense -> GELU -> Drop -> Dense -> Drop.

    Args:
        inputs: Input token tensor of shape ``(B, N, D)``.
        hidden_features: Hidden expansion dimension of the first Dense.
        out_features: Output dimension of the second Dense.
        drop: Dropout rate applied after each Dense.
        block_idx: Numeric index used to name the inner layers
            (``blocks_{block_idx}_*``).

    Returns:
        Tensor of shape ``(B, N, out_features)``.
    """
    x = layers.Dense(
        hidden_features, use_bias=True, name=f"blocks_{block_idx}_dense_1"
    )(inputs)
    x = layers.Activation("gelu", name=f"blocks_{block_idx}_gelu")(x)
    x = layers.Dropout(drop, name=f"blocks_{block_idx}_dropout_1")(x)
    x = layers.Dense(out_features, use_bias=True, name=f"blocks_{block_idx}_dense_2")(x)
    x = layers.Dropout(drop, name=f"blocks_{block_idx}_dropout_2")(x)
    return x


def transformer_block(
    inputs,
    dim,
    num_heads,
    mlp_ratio=4.0,
    qkv_bias=False,
    qk_norm=False,
    proj_drop=0.0,
    attn_drop=0.0,
    block_idx=0,
    init_values=None,
):
    """Standard ViT transformer block: LN -> MHSA -> Add -> LN -> MLP -> Add.

    Args:
        inputs: Input token tensor of shape ``(B, N, dim)``.
        dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden expansion ratio for the MLP sub-block.
        qkv_bias: Whether to include bias in the QKV projection.
        qk_norm: Whether to apply LayerNorm to Q and K inside attention.
        proj_drop: Dropout rate on the attention output projection and MLP.
        attn_drop: Dropout rate applied to attention weights.
        block_idx: Numeric index used to name layers inside this block.
        init_values: If set, apply LayerScale with this initial gamma on
            both residual branches.

    Returns:
        Tensor of shape ``(B, N, dim)`` after both residual branches.
    """
    x = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=f"blocks_{block_idx}_layernorm_1"
    )(inputs)
    x = MultiHeadSelfAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        block_prefix=f"blocks_{block_idx}",
    )(x)
    if init_values:
        x = LayerScale(
            init_values=init_values, name=f"blocks_{block_idx}_layerscale_1"
        )(x)
    x = keras.layers.Add(name=f"blocks_{block_idx}_add_1")([x, inputs])

    y = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=f"blocks_{block_idx}_layernorm_2"
    )(x)
    y = mlp_block(
        y,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        drop=proj_drop,
        block_idx=block_idx,
    )
    if init_values:
        y = LayerScale(
            init_values=init_values, name=f"blocks_{block_idx}_layerscale_2"
        )(y)
    return keras.layers.Add(name=f"blocks_{block_idx}_add_2")([x, y])


def vit_backbone_feature(
    inputs,
    *,
    patch_size,
    dim,
    depth,
    num_heads,
    mlp_ratio,
    qkv_bias,
    qk_norm,
    drop_rate,
    attn_drop_rate,
    no_embed_class,
    use_distillation,
    init_values,
    image_size,
    data_format,
    return_intermediates=False,
    return_stages=False,
):
    """ViT patch embed + cls/dist tokens + pos embed + transformer blocks.

    Shared by :class:`ViTClassify` and :class:`ViTModel`.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        patch_size: Conv-stem patch size in pixels.
        dim: Token embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads per block.
        mlp_ratio: Hidden expansion ratio for the MLP sub-block.
        qkv_bias: Whether to include bias in the QKV projection.
        qk_norm: Whether to apply LayerNorm to Q and K inside attention.
        drop_rate: Dropout rate after pos-embed and inside the MLP.
        attn_drop_rate: Dropout rate applied to attention weights.
        no_embed_class: If ``True``, position embeddings do not cover the
            class/distillation prefix tokens.
        use_distillation: If ``True``, prepend a separate distillation token
            in addition to the class token.
        init_values: Optional LayerScale initial gamma value.
        image_size: Input image resolution; used when ``inputs`` has unknown
            spatial shape.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_intermediates: If ``True``, return per-block raw outputs
            (no final LN) — used by DINO / DINOv2 which apply their own norm.
        return_stages: If ``True``, return a list of per-block outputs ending
            with the final-LN output (used as a generic backbone feature
            extractor).

    Returns:
        Final encoder tokens of shape ``(B, num_tokens, dim)`` after the
        final LayerNorm. When ``return_intermediates=True``, a list
        ``[post_pos_embed, block_0, ..., block_{depth-1}]`` of raw block
        outputs (no final LN) is returned instead. When ``return_stages=True``,
        a list ``[block_0, ..., block_{depth-1}, final_ln]`` is returned.
    """
    if data_format == "channels_first":
        _, height, width = inputs.shape[1:]
    else:
        height, width, _ = inputs.shape[1:]
    if height is None:
        height = image_size
    if width is None:
        width = image_size
    grid_h = height // patch_size
    grid_w = width // patch_size

    x = layers.Conv2D(
        filters=dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        data_format=data_format,
        name="conv1",
    )(inputs)
    x = layers.Reshape((-1, dim))(x)
    x = ClassDistToken(use_distillation=use_distillation, name="cls_token")(x)
    x = AddPositionEmbs(
        name="pos_embed",
        no_embed_class=no_embed_class,
        use_distillation=use_distillation,
        grid_h=grid_h,
        grid_w=grid_w,
    )(x)
    intermediates = [x]
    x = layers.Dropout(drop_rate)(x)

    stages = []
    for i in range(depth):
        x = transformer_block(
            x,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=drop_rate,
            attn_drop=attn_drop_rate,
            init_values=init_values,
            block_idx=i,
        )
        intermediates.append(x)
        stages.append(x)

    if return_intermediates:
        return intermediates
    x = layers.LayerNormalization(epsilon=1e-6, axis=-1, name="final_layernorm")(x)
    stages.append(x)
    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class ViTModel(BaseModel):
    """ViT backbone — the main feature extractor.

    Returns the final-LN normalized token sequence ``(B, num_tokens, dim)``.
    The first 1 (or 2 if distillation) tokens are class/distillation tokens;
    the rest are spatial patch tokens. This is the last layer output before
    the classifier head. :class:`ViTClassify` composes this model and reads
    the class token(s) to produce logits.

    Reference:
    - [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

    Construction:

    >>> ViTModel.from_weights("vit_base_patch16_224_augreg_in21k_ft_in1k")
    >>> ViTModel.from_weights("timm:timm/vit_base_patch16_224.augreg_in21k_ft_in1k")
    """

    KMODELS_CONFIG = VIT_CONFIG
    KMODELS_WEIGHTS = VIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = ViTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_vit_weights(keras_model, state_dict)

    def __init__(
        self,
        as_backbone=False,
        patch_size=16,
        dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        no_embed_class=False,
        use_distillation=False,
        init_values=None,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="ViTModel",
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
        x = vit_backbone_feature(
            x,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            no_embed_class=no_embed_class,
            use_distillation=use_distillation,
            init_values=init_values,
            image_size=image_size,
            data_format=data_format,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.as_backbone = as_backbone
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.no_embed_class = no_embed_class
        self.use_distillation = use_distillation
        self.init_values = init_values
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
                "dim": self.dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "qk_norm": self.qk_norm,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "no_embed_class": self.no_embed_class,
                "use_distillation": self.use_distillation,
                "init_values": self.init_values,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class ViTClassify(BaseModel):
    """Vision Transformer classifier — :class:`ViTModel` + linear head on the CLS token.

    Wraps a :class:`ViTModel` backbone and attaches a single Dense layer
    on the CLS token (index 0 of the backbone's output) to produce class
    logits. When ``use_distillation`` is True, two prediction heads are
    used (one on the CLS token, one on the distillation token) and their
    outputs are averaged.

    Reference:
    - [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

    Construction:

    >>> ViTClassify.from_weights("vit_base_patch16_224_augreg_in21k_ft_in1k")
    >>> ViTClassify.from_weights("timm:timm/vit_base_patch16_224.augreg_in21k_ft_in1k")
    """

    KMODELS_CONFIG = VIT_CONFIG
    KMODELS_WEIGHTS = VIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_vit_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        no_embed_class=False,
        use_distillation=False,
        init_values=None,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ViTClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        backbone = ViTModel(
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            no_embed_class=no_embed_class,
            use_distillation=use_distillation,
            init_values=init_values,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = backbone.output
        if use_distillation:
            cls_token = layers.Lambda(lambda v: v[:, 0], name="ExtractClsToken")(x)
            dist_token = layers.Lambda(lambda v: v[:, 1], name="ExtractDistToken")(x)
            cls_token = layers.Dropout(drop_rate)(cls_token)
            dist_token = layers.Dropout(drop_rate)(dist_token)
            cls_head = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(cls_token)
            dist_head = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions_dist",
            )(dist_token)
            out = (cls_head + dist_head) / 2
        else:
            tok = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
            tok = layers.Dropout(drop_rate)(tok)
            out = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(tok)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.no_embed_class = no_embed_class
        self.use_distillation = use_distillation
        self.init_values = init_values
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
                "dim": self.dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "qk_norm": self.qk_norm,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "no_embed_class": self.no_embed_class,
                "use_distillation": self.use_distillation,
                "init_values": self.init_values,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
