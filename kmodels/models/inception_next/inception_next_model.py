import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer, LayerScale
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import INCEPTION_NEXT_CONFIG, INCEPTION_NEXT_WEIGHTS
from .convert_inception_next_torch_to_keras import transfer_inception_next_weights


def inception_dwconv2d(
    x,
    square_kernel_size=3,
    band_kernel_size=11,
    branch_ratio=0.125,
    data_format=None,
    channels_axis=None,
    name="token_mixer",
):
    """Inception-style token mixer: square + band depthwise convs over channel splits."""
    input_channels = x.shape[channels_axis]
    branch_channels = int(input_channels * branch_ratio)
    split_sizes = [input_channels - 3 * branch_channels] + [branch_channels] * 3
    split_indices = [sum(split_sizes[: i + 1]) for i in range(len(split_sizes) - 1)]

    def calculate_padding(kernel_size):
        return (kernel_size - 1) // 2

    square_padding, band_padding = (
        calculate_padding(square_kernel_size),
        calculate_padding(band_kernel_size),
    )

    x_splits = keras.ops.split(x, split_indices, axis=channels_axis)
    x_id, *x_branches = x_splits

    conv_configs = [
        (square_kernel_size, square_padding, f"{name}_dwconv_hw"),
        ((1, band_kernel_size), (0, band_padding), f"{name}_dwconv_w"),
        ((band_kernel_size, 1), (band_padding, 0), f"{name}_dwconv_h"),
    ]

    x = [
        layers.DepthwiseConv2D(
            kernel, use_bias=True, data_format=data_format, name=lname
        )(layers.ZeroPadding2D(padding)(branch_input))
        for (kernel, padding, lname), branch_input in zip(conv_configs, x_branches)
    ]

    return layers.Concatenate(axis=channels_axis)([x_id, *x])


def inception_next_block(
    x,
    num_filter,
    mlp_ratio=4.0,
    dropout_rate=0.0,
    layer_scale_init_value=1e-6,
    band_kernel_size=11,
    branch_ratio=0.125,
    data_format=None,
    channels_axis=None,
    name="blocks",
):
    """InceptionNeXt block: token mixer -> BN -> Conv MLP -> LayerScale -> residual."""
    x_input = x

    x = inception_dwconv2d(
        x,
        band_kernel_size=band_kernel_size,
        branch_ratio=branch_ratio,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_token_mixer",
    )

    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, name=f"{name}_batchnorm"
    )(x)

    x = layers.Conv2D(
        int(num_filter * mlp_ratio),
        1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv1",
    )(x)
    x = layers.Activation("gelu", name=f"{name}_act")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(
        num_filter, 1, use_bias=True, data_format=data_format, name=f"{name}_conv2"
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    x = LayerScale(layer_scale_init_value, name=f"{name}_gamma")(x)
    x = layers.Add()([x, x_input])

    return x


def _inception_next_features(
    inputs,
    *,
    depths,
    num_filters,
    mlp_ratios,
    band_kernel_size,
    branch_ratio,
    data_format,
    channels_axis,
):
    """InceptionNeXt stem + 4 stages, returns ``[stem, s1, s2, s3, s4]``."""
    features = []

    x = layers.Conv2D(
        num_filters[0],
        4,
        4,
        use_bias=True,
        data_format=data_format,
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_batchnorm"
    )(x)
    features.append(x)

    for i in range(len(depths)):
        strides = 2 if i > 0 else 1
        if strides > 1:
            x = layers.BatchNormalization(
                axis=channels_axis,
                momentum=0.9,
                epsilon=1e-5,
                name=f"stages_{i}_downsample_batchnorm",
            )(x)
            x = layers.Conv2D(
                num_filters[i],
                2,
                strides,
                use_bias=True,
                data_format=data_format,
                name=f"stages_{i}_downsample_conv",
            )(x)

        for j in range(depths[i]):
            x = inception_next_block(
                x,
                num_filter=num_filters[i],
                mlp_ratio=mlp_ratios[i],
                band_kernel_size=band_kernel_size,
                branch_ratio=branch_ratio,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"stages_{i}_blocks_{j}",
            )
        features.append(x)

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class InceptionNext(BaseModel):
    """InceptionNeXt classifier (timm-ported).

    Reference:
    - [InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/abs/2303.16900)

    Construction:

    >>> InceptionNext.from_weights("inception_next_tiny_sail_in1k")
    >>> InceptionNext.from_weights("timm:timm/inception_next_tiny.sail_in1k")
    """

    KMODELS_CONFIG = INCEPTION_NEXT_CONFIG
    KMODELS_WEIGHTS = INCEPTION_NEXT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_inception_next_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        num_filters=(96, 192, 384, 768),
        mlp_ratios=(4, 4, 4, 3),
        band_kernel_size=11,
        branch_ratio=0.125,
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="InceptionNext",
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
        features = _inception_next_features(
            x,
            depths=depths,
            num_filters=num_filters,
            mlp_ratios=mlp_ratios,
            band_kernel_size=band_kernel_size,
            branch_ratio=branch_ratio,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.Dense(int(num_filters[-1] * 3.0), use_bias=True, name="head_fc")(x)
        x = layers.Activation("gelu")(x)
        x = layers.LayerNormalization(epsilon=1e-6, name="head_batchnorm")(x)
        x = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depths = list(depths)
        self.num_filters = list(num_filters)
        self.mlp_ratios = list(mlp_ratios)
        self.band_kernel_size = band_kernel_size
        self.branch_ratio = branch_ratio
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
                "depths": self.depths,
                "num_filters": self.num_filters,
                "mlp_ratios": self.mlp_ratios,
                "band_kernel_size": self.band_kernel_size,
                "branch_ratio": self.branch_ratio,
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
class InceptionNextBackbone(BaseModel):
    """InceptionNeXt feature extractor. Returns ``[stem, s1, s2, s3, s4]``."""

    KMODELS_CONFIG = INCEPTION_NEXT_CONFIG
    KMODELS_WEIGHTS = INCEPTION_NEXT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        return InceptionNext

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
        transfer_inception_next_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        num_filters=(96, 192, 384, 768),
        mlp_ratios=(4, 4, 4, 3),
        band_kernel_size=11,
        branch_ratio=0.125,
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        name="InceptionNextBackbone",
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
            require_flatten=False,
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
        features = _inception_next_features(
            x,
            depths=depths,
            num_filters=num_filters,
            mlp_ratios=mlp_ratios,
            band_kernel_size=band_kernel_size,
            branch_ratio=branch_ratio,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.depths = list(depths)
        self.num_filters = list(num_filters)
        self.mlp_ratios = list(mlp_ratios)
        self.band_kernel_size = band_kernel_size
        self.branch_ratio = branch_ratio
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "num_filters": self.num_filters,
                "mlp_ratios": self.mlp_ratios,
                "band_kernel_size": self.band_kernel_size,
                "branch_ratio": self.branch_ratio,
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
