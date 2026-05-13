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
    """Standard transformer MLP block (Dense -> GELU -> Drop -> Dense -> Drop)."""
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
    """LN -> MHSA -> Add -> LN -> MLP -> Add."""
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


def _vit_features(
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
):
    """ViT patch embed + cls/dist tokens + pos embed + transformer blocks.

    Returns the final encoder tokens (post final LayerNorm), shape
    ``(B, num_tokens, dim)``. Shared by :class:`ViT` and :class:`ViTBackbone`.

    When ``return_intermediates=True``, returns a list ``[post_pos_embed,
    block_0, ..., block_{depth-1}]`` of per-block raw outputs (no final
    LN). Used by DINO / DINOv2 which apply their own normalization.
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

    if return_intermediates:
        return intermediates
    return layers.LayerNormalization(epsilon=1e-6, axis=-1, name="final_layernorm")(x)


@keras.saving.register_keras_serializable(package="kmodels")
class ViT(BaseModel):
    """
    Vision Transformer classifier (timm-ported).

    Reference:
    - [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

    Construction:

    >>> ViT.from_weights("vit_base_patch16_224_augreg_in21k_ft_in1k")
    >>> ViT.from_weights("timm:timm/vit_base_patch16_224.augreg_in21k_ft_in1k")
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
        name="ViT",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

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
        x = _vit_features(
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
        )

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
            x = (cls_head + dist_head) / 2
        else:
            x = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
            x = layers.Dropout(drop_rate)(x)
            x = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

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


@keras.saving.register_keras_serializable(package="kmodels")
class ViTBackbone(BaseModel):
    """ViT feature extractor (no classifier head).

    Returns the final encoder tokens (post final LayerNorm), shape
    ``(B, num_tokens, dim)``. The first 1 (or 2 if distillation) tokens are
    class/distillation tokens; the rest are spatial patch tokens.

    Construction:

    >>> ViTBackbone.from_weights("vit_base_patch16_224_augreg_in21k_ft_in1k")
    >>> ViTBackbone.from_weights("timm:timm/vit_base_patch16_224.augreg_in21k_ft_in1k")
    """

    KMODELS_CONFIG = VIT_CONFIG
    KMODELS_WEIGHTS = VIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        return ViT

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = cls._release_warm_start_cls().from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

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
        name="ViTBackbone",
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
        x = _vit_features(
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
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

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
