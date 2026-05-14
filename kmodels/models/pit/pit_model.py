import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.models.vit.vit_layers import (
    AddPositionEmbs,
    ClassDistToken,
    MultiHeadSelfAttention,
)
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import PIT_CONFIG, PIT_WEIGHTS
from .convert_pit_torch_to_keras import transfer_pit_weights


def mlp_block(inputs, hidden_features, out_features=None, drop=0.0, block_prefix=None):
    """Standard transformer MLP block."""
    x = layers.Dense(hidden_features, use_bias=True, name=block_prefix + "_dense_1")(
        inputs
    )
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(out_features, use_bias=True, name=block_prefix + "_dense_2")(x)
    x = layers.Dropout(drop)(x)
    return x


def transformer_block(inputs, dim, num_heads, mlp_ratio, block_prefix=None):
    """LN -> MHSA -> Add -> LN -> MLP -> Add."""
    x = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=block_prefix + "_layernorm_1"
    )(inputs)
    x = MultiHeadSelfAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        block_prefix=block_prefix.replace("pit", "transformers"),
    )(x)
    x = layers.Add()([inputs, x])

    y = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=block_prefix + "_layernorm_2"
    )(x)
    y = mlp_block(
        y,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        block_prefix=block_prefix,
    )
    return layers.Add()([x, y])


def conv_pooling(
    x, nb_tokens, in_channels, out_channels, stride, data_format, block_prefix
):
    """Depthwise-conv downsampling for spatial tokens + Dense projection for class tokens."""
    input_tensor, (height, width) = x
    tokens = input_tensor[:, :nb_tokens]
    spatial = input_tensor[:, nb_tokens:]

    new_height = (height + stride - 1) // stride
    new_width = (width + stride - 1) // stride

    spatial = layers.Reshape((height, width, in_channels))(spatial)
    if data_format == "channels_first":
        spatial = layers.Permute((3, 1, 2))(spatial)
    spatial = layers.ZeroPadding2D(data_format=data_format, padding=stride // 2)(
        spatial
    )
    spatial = layers.Conv2D(
        filters=out_channels,
        kernel_size=stride + 1,
        strides=stride,
        groups=in_channels,
        data_format=data_format,
        name=block_prefix + "_conv",
    )(spatial)

    tokens = layers.Dense(units=out_channels, name=block_prefix + "_dense")(tokens)
    if data_format == "channels_first":
        spatial = layers.Permute((2, 3, 1))(spatial)
    spatial = layers.Reshape((new_height * new_width, out_channels))(spatial)
    output = layers.Concatenate(axis=1)([tokens, spatial])
    return output, (new_height, new_width)


def _pit_features(
    inputs,
    *,
    patch_size,
    stride,
    embed_dim,
    depth,
    heads,
    mlp_ratio,
    distilled,
    drop_rate,
    image_size,
    data_format,
    return_stages=False,
    return_final_spatial=False,
):
    """PiT stem + pooling-attention stages.

    Returns the final pre-norm tokens (cls + dist if distilled), already
    sliced to the class/dist tokens. When ``return_stages=True``, returns
    a list ``[stem, stage1, stage2, ..., stageN, cls_dist_norm]``. When
    ``return_final_spatial=True``, returns the final stage's spatial feature
    map of shape ``(B, H, W, C)`` (or channels_first equivalent).
    """
    if data_format == "channels_first":
        _, height, width = inputs.shape[1:]
    else:
        height, width, _ = inputs.shape[1:]

    x = layers.Conv2D(
        filters=embed_dim[0],
        kernel_size=patch_size,
        strides=stride,
        data_format=data_format,
        name="patch_embed_conv",
    )(inputs)

    grid_h = (height - patch_size) // stride + 1
    grid_w = (width - patch_size) // stride + 1
    input_size = (grid_h, grid_w)

    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1), name="patch_to_nhwc")(x)
    x = layers.Reshape((grid_h * grid_w, embed_dim[0]), name="patch_tokens_reshape")(x)

    x = AddPositionEmbs(
        grid_h=grid_h,
        grid_w=grid_w,
        no_embed_class=True,
        use_distillation=distilled,
        name="pos_embed",
    )(x)
    x = ClassDistToken(
        use_distillation=distilled,
        combine_tokens=True,
        name="class_dist_token",
    )(x)

    stages = [x]
    x = layers.Dropout(drop_rate, name="pos_drop")(x)

    for stage_idx in range(len(depth)):
        for block_idx in range(depth[stage_idx]):
            x = transformer_block(
                x,
                dim=embed_dim[stage_idx],
                num_heads=heads[stage_idx],
                mlp_ratio=mlp_ratio,
                block_prefix=f"pit_{stage_idx}_blocks_{block_idx}",
            )
        if stage_idx < len(depth) - 1:
            x, input_size = conv_pooling(
                (x, input_size),
                nb_tokens=2 if distilled else 1,
                in_channels=embed_dim[stage_idx],
                out_channels=embed_dim[stage_idx + 1],
                stride=2,
                data_format=data_format,
                block_prefix=f"pit_{stage_idx + 1}_pool",
            )
        stages.append(x)

    cls_dist = x[:, : 2 if distilled else 1]
    cls_dist = layers.LayerNormalization(epsilon=1e-6, axis=-1, name="norm")(cls_dist)

    if return_final_spatial:
        nb_tokens = 2 if distilled else 1
        final_channels = embed_dim[-1]
        spatial = x[:, nb_tokens:]
        spatial = layers.Reshape(
            (input_size[0], input_size[1], final_channels),
            name="final_spatial_reshape",
        )(spatial)
        if data_format == "channels_first":
            spatial = layers.Permute((3, 1, 2), name="final_spatial_to_cf")(spatial)
        return spatial

    if return_stages:
        stages.append(cls_dist)
        return stages
    return cls_dist


@keras.saving.register_keras_serializable(package="kmodels")
class PiTClassify(BaseModel):
    """Pooling-based Vision Transformer classifier (timm-ported).

    Reference:
    - [Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302)

    Construction:

    >>> PiTClassify.from_weights("pit_b_224_in1k")
    >>> PiTClassify.from_weights("timm:timm/pit_b_224.in1k")
    """

    KMODELS_CONFIG = PIT_CONFIG
    KMODELS_WEIGHTS = PIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_pit_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        stride=8,
        embed_dim=(64, 128, 256),
        depth=(2, 6, 4),
        heads=(2, 4, 8),
        mlp_ratio=4.0,
        distilled=False,
        drop_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        num_classes=1000,
        classifier_activation="linear",
        name="PiTClassify",
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
        cls_dist = _pit_features(
            x,
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            distilled=distilled,
            drop_rate=drop_rate,
            image_size=image_size,
            data_format=data_format,
        )

        if distilled:
            cls_token = layers.Lambda(lambda v: v[:, 0], name="ExtractClsToken")(
                cls_dist
            )
            dist_token = layers.Lambda(lambda v: v[:, 1], name="ExtractDistToken")(
                cls_dist
            )
            cls_token = layers.Dropout(drop_rate)(cls_token)
            dist_token = layers.Dropout(drop_rate)(dist_token)
            cls_head = layers.Dense(num_classes, name="predictions")(cls_token)
            dist_head = layers.Dense(num_classes, name="predictions_dist")(dist_token)
            out = layers.Average()([cls_head, dist_head])
            if classifier_activation is not None:
                out = layers.Activation(
                    classifier_activation, name="predictions_activation"
                )(out)
        else:
            tok = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(cls_dist)
            tok = layers.Dropout(drop_rate)(tok)
            out = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(tok)

        super().__init__(inputs=img_input, outputs=out, name=name, **kwargs)

        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.distilled = distilled
        self.drop_rate = drop_rate
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
                "stride": self.stride,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "heads": self.heads,
                "mlp_ratio": self.mlp_ratio,
                "distilled": self.distilled,
                "drop_rate": self.drop_rate,
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
class PiTBackbone(BaseModel):
    """PiT feature extractor (no classifier head). Returns final norm'd cls/dist tokens."""

    KMODELS_CONFIG = PIT_CONFIG
    KMODELS_WEIGHTS = PIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = PiTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_pit_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        stride=8,
        embed_dim=(64, 128, 256),
        depth=(2, 6, 4),
        heads=(2, 4, 8),
        mlp_ratio=4.0,
        distilled=False,
        drop_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        name="PiTBackbone",
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
        cls_dist = _pit_features(
            x,
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            distilled=distilled,
            drop_rate=drop_rate,
            image_size=image_size,
            data_format=data_format,
        )

        super().__init__(inputs=img_input, outputs=cls_dist, name=name, **kwargs)

        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.distilled = distilled
        self.drop_rate = drop_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "stride": self.stride,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "heads": self.heads,
                "mlp_ratio": self.mlp_ratio,
                "distilled": self.distilled,
                "drop_rate": self.drop_rate,
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
class PiTModel(BaseModel):
    """PiT trunk returning the final stage spatial feature map ``(B, H, W, C)``."""

    KMODELS_CONFIG = PIT_CONFIG
    KMODELS_WEIGHTS = PIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = PiTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_pit_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        stride=8,
        embed_dim=(64, 128, 256),
        depth=(2, 6, 4),
        heads=(2, 4, 8),
        mlp_ratio=4.0,
        distilled=False,
        drop_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        name="PiTModel",
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
        spatial = _pit_features(
            x,
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            distilled=distilled,
            drop_rate=drop_rate,
            image_size=image_size,
            data_format=data_format,
            return_final_spatial=True,
        )

        super().__init__(inputs=img_input, outputs=spatial, name=name, **kwargs)

        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.distilled = distilled
        self.drop_rate = drop_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "stride": self.stride,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "heads": self.heads,
                "mlp_ratio": self.mlp_ratio,
                "distilled": self.distilled,
                "drop_rate": self.drop_rate,
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
