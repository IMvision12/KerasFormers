import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer, LayerScale, StochasticDepth
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import POOLFORMER_CONFIG, POOLFORMER_WEIGHTS
from .convert_poolformer_torch_to_keras import transfer_poolformer_weights


def mlp_block(x, hidden_dim, embed_dim, drop_rate, data_format, name):
    """
    MLP block using 1x1 convolutions for vision models.

    Args:
        x: Input tensor to the MLP block.
        hidden_dim: Number of filters in the first (hidden) convolution layer.
        embed_dim: Number of filters in the second (output) convolution layer.
        drop_rate: Dropout rate applied after each convolution layer.
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        name: Base name for the layers in this block, used for layer naming.

    Returns:
        Output tensor after passing through the MLP block.
    """
    x = layers.Conv2D(
        filters=hidden_dim,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_1",
    )(x)
    x = layers.Activation("gelu", name=f"{name}_act")(x)
    x = layers.Dropout(drop_rate, name=f"{name}_drop_1")(x)

    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_2",
    )(x)
    x = layers.Dropout(drop_rate, name=f"{name}_drop_2")(x)

    return x


def poolformer_block(
    x,
    embed_dim,
    mlp_ratio,
    drop_rate,
    drop_path_rate,
    init_scale,
    data_format,
    channels_axis,
    name,
):
    """PoolFormer block (pooling-based token mixer + ConvMLP, with LayerScale)."""
    shortcut = x

    x = layers.GroupNormalization(
        groups=1, axis=channels_axis, epsilon=1e-5, name=f"{name}_groupnorm_1"
    )(x)

    x_pool = layers.AveragePooling2D(
        pool_size=3, strides=1, padding="same", data_format=data_format
    )(x)
    x = layers.Subtract(name=f"{name}_token_mixer")([x_pool, x])

    layer_scale_1 = LayerScale(init_scale, name=f"{name}_layerscale_1")(x)

    if drop_path_rate > 0:
        layer_scale_1 = StochasticDepth(drop_path_rate)(layer_scale_1)

    x = layers.Add(name=f"{name}_add_1")([shortcut, layer_scale_1])

    shortcut = x
    x = layers.GroupNormalization(
        groups=1, axis=channels_axis, epsilon=1e-5, name=f"{name}_groupnorm_2"
    )(x)
    x = mlp_block(
        x,
        hidden_dim=int(embed_dim * mlp_ratio),
        embed_dim=embed_dim,
        drop_rate=drop_rate,
        data_format=data_format,
        name=f"{name}_mlp",
    )

    layer_scale_2 = LayerScale(init_scale, name=f"{name}_layerscale_2")(x)

    if drop_path_rate > 0:
        layer_scale_2 = StochasticDepth(drop_path_rate)(layer_scale_2)

    x = layers.Add(name=f"{name}_add_2")([shortcut, layer_scale_2])

    return x


def _poolformer_features(
    inputs,
    *,
    embed_dims,
    num_blocks,
    mlp_ratio,
    drop_rate,
    drop_path_rate,
    init_scale,
    data_format,
    channels_axis,
):
    """Stem + 4 stages of PoolFormer blocks, returns ``[stem, s1..s4]`` (5 maps)."""
    features = []

    x = layers.ZeroPadding2D(
        padding=((2, 2), (2, 2)), data_format=data_format, name="stem_pad"
    )(inputs)
    x = layers.Conv2D(
        filters=embed_dims[0],
        kernel_size=7,
        strides=4,
        use_bias=True,
        padding="valid",
        data_format=data_format,
        name="stem_conv",
    )(x)
    features.append(x)

    total_blocks = sum(num_blocks)
    dpr = [val * drop_path_rate / total_blocks for val in range(total_blocks)]
    cur = 0

    for stage_idx in range(len(num_blocks)):
        for block_idx in range(num_blocks[stage_idx]):
            x = poolformer_block(
                x,
                embed_dim=embed_dims[stage_idx],
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path_rate=dpr[cur],
                init_scale=init_scale,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"stage_{stage_idx}_block_{block_idx}",
            )
            cur += 1

        features.append(x)

        if stage_idx < len(num_blocks) - 1:
            x = layers.ZeroPadding2D(
                padding=((1, 1), (1, 1)),
                data_format=data_format,
                name=f"stage_{stage_idx + 1}_downsample_pad",
            )(x)
            x = layers.Conv2D(
                filters=embed_dims[stage_idx + 1],
                kernel_size=3,
                strides=2,
                use_bias=True,
                padding="valid",
                data_format=data_format,
                name=f"stage_{stage_idx + 1}_downsample_conv",
            )(x)

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class PoolFormerClassify(BaseModel):
    """PoolFormer classifier (timm-ported).

    Reference:
    - [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)

    Construction:

    >>> PoolFormerClassify.from_weights("poolformer_s12_sail_in1k")
    >>> PoolFormerClassify.from_weights("timm:timm/poolformer_s12.sail_in1k")
    """

    KMODELS_CONFIG = POOLFORMER_CONFIG
    KMODELS_WEIGHTS = POOLFORMER_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_poolformer_weights(keras_model, state_dict)

    def __init__(
        self,
        embed_dims=(64, 128, 320, 512),
        num_blocks=(2, 2, 6, 2),
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        init_scale=1e-5,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="PoolFormerClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

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
        features = _poolformer_features(
            x,
            embed_dims=embed_dims,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            init_scale=init_scale,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.LayerNormalization(epsilon=1e-6, name="layernorm")(x)
        x = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.embed_dims = embed_dims
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_scale = init_scale
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
                "embed_dims": self.embed_dims,
                "num_blocks": self.num_blocks,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "init_scale": self.init_scale,
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
class PoolFormerBackbone(BaseModel):
    """PoolFormer feature extractor. Returns ``[stem, s1..s4]`` (5 maps)."""

    KMODELS_CONFIG = POOLFORMER_CONFIG
    KMODELS_WEIGHTS = POOLFORMER_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = PoolFormerClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_poolformer_weights(keras_model, state_dict)

    def __init__(
        self,
        embed_dims=(64, 128, 320, 512),
        num_blocks=(2, 2, 6, 2),
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        init_scale=1e-5,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="PoolFormerBackbone",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

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
        features = _poolformer_features(
            x,
            embed_dims=embed_dims,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            init_scale=init_scale,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.embed_dims = embed_dims
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_scale = init_scale
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dims": self.embed_dims,
                "num_blocks": self.num_blocks,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "init_scale": self.init_scale,
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
class PoolFormerModel(BaseModel):
    """PoolFormer trunk returning the final stage feature map ``(B, H, W, C)``."""

    KMODELS_CONFIG = POOLFORMER_CONFIG
    KMODELS_WEIGHTS = POOLFORMER_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = PoolFormerClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_poolformer_weights(keras_model, state_dict)

    def __init__(
        self,
        embed_dims=(64, 128, 320, 512),
        num_blocks=(2, 2, 6, 2),
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        init_scale=1e-5,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="PoolFormerModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

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
        features = _poolformer_features(
            x,
            embed_dims=embed_dims,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            init_scale=init_scale,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features[-1], name=name, **kwargs)

        self.embed_dims = embed_dims
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_scale = init_scale
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dims": self.embed_dims,
                "num_blocks": self.num_blocks,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "init_scale": self.init_scale,
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
