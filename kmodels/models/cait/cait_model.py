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

from .config import CAIT_CONFIG, CAIT_WEIGHTS
from .convert_cait_torch_to_keras import transfer_cait_weights


def mlp_block(x, hidden_dim, out_dim, drop_rate=0.0, block_prefix=None):
    """MLP block: Dense -> GELU -> Drop -> Dense -> Drop."""
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


def _layer_scale_talking_head_block(
    x,
    embed_dim,
    num_heads,
    mlp_ratio=4.0,
    drop_rate=0.0,
    init_values=1e-5,
    block_prefix="block",
):
    """LN -> TalkingHeadAttn -> LayerScale -> SD -> Add -> LN -> MLP -> LayerScale -> SD -> Add."""
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


def _layer_scale_class_attn_block(
    cls_token,
    x,
    embed_dim,
    num_heads,
    mlp_ratio=4.0,
    init_values=1e-5,
    block_prefix="block_token_only",
):
    """Class-attention-only block that updates cls_token using patch tokens."""
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


def _cait_features(
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
):
    """CaiT stem + talking-head blocks + class-attn blocks. Returns final norm'd tokens."""
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

    dpr = list(ops.linspace(0.0, drop_path_rate, depth))
    for i in range(depth):
        x = _layer_scale_talking_head_block(
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=dpr[i],
            init_values=1e-5,
            block_prefix=f"blocks_{i}",
        )

    cls_token = ClassDistToken(name="cls_token")(x)
    for i in range(depth_token_only):
        cls_token = _layer_scale_class_attn_block(
            cls_token,
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            init_values=1e-5,
            block_prefix=f"blocks_token_only_{i}",
        )

    x = layers.Concatenate(axis=1, name="cat_cls_patch")([cls_token, x])
    return layers.LayerNormalization(epsilon=1e-6, name="final_layernorm")(x)


@keras.saving.register_keras_serializable(package="kmodels")
class CaiTClassify(BaseModel):
    """Class-Attention in Image Transformers classifier (timm-ported).

    Reference:
    - [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239)

    Construction:

    >>> CaiTClassify.from_weights("cait_s24_224_fb_dist_in1k")
    >>> CaiTClassify.from_weights("timm:timm/cait_s24_224.fb_dist_in1k")
    """

    KMODELS_CONFIG = CAIT_CONFIG
    KMODELS_WEIGHTS = CAIT_WEIGHTS
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
        x = _cait_features(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            data_format=data_format,
        )

        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x[:, 0])

        super().__init__(inputs=img_input, outputs=out, name=name, **kwargs)

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


@keras.saving.register_keras_serializable(package="kmodels")
class CaiTBackbone(BaseModel):
    """CaiT feature extractor (no classifier head). Returns final norm'd tokens."""

    KMODELS_CONFIG = CAIT_CONFIG
    KMODELS_WEIGHTS = CAIT_WEIGHTS
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
        name="CaiTBackbone",
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
        x = _cait_features(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            data_format=data_format,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

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
class CaiTModel(BaseModel):
    """CaiT trunk returning the final feature as a 4D map.

    Drops the cls token from the final encoder output and reshapes the
    remaining patch tokens to ``(B, H, W, D)``. For raw tokens use
    :class:`CaiTBackbone`; for class logits use :class:`CaiTClassify`.
    """

    KMODELS_CONFIG = CAIT_CONFIG
    KMODELS_WEIGHTS = CAIT_WEIGHTS
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

        if data_format == "channels_first":
            _, h_in, w_in = input_shape
        else:
            h_in, w_in, _ = input_shape
        grid_h = h_in // patch_size
        grid_w = w_in // patch_size

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(img_input)
            if include_normalization
            else img_input
        )
        x = _cait_features(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            data_format=data_format,
        )

        patches = layers.Lambda(lambda v: v[:, 1:], name="drop_prefix_tokens")(x)
        feat = layers.Reshape((grid_h, grid_w, embed_dim), name="tokens_to_grid")(
            patches
        )

        super().__init__(inputs=img_input, outputs=feat, name=name, **kwargs)

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
