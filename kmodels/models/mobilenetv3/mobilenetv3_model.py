import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import MOBILENETV3_CONFIG, MOBILENETV3_WEIGHTS
from .convert_mobilenetv3_keras_to_keras import transfer_mobilenetv3_weights


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """Snap a (possibly scaled) channel count to a multiple of ``divisor``."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def inverted_residual_block(
    x,
    expansion_ratio,
    filters,
    kernel_size,
    stride,
    se_ratio,
    activation,
    block_id,
    data_format,
    channels_axis,
):
    """MobileNetV3-style inverted residual block (with optional SE)."""
    shortcut = x
    prefix = f"ir_block_{block_id}"
    input_filters = x.shape[channels_axis]
    expanded_filters = make_divisible(input_filters * expansion_ratio)

    if expansion_ratio != 1:
        x = layers.Conv2D(
            expanded_filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=f"{prefix}_conv_pw",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=f"{prefix}_batchnorm_1",
        )(x)
        x = layers.Activation(activation, name=f"{prefix}_activation_1")(x)

    if stride == 1:
        pad_h = pad_w = kernel_size // 2
        x = layers.ZeroPadding2D(data_format=data_format, padding=(pad_h, pad_w))(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_dwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=f"{prefix}_batchnorm_2",
    )(x)
    x = layers.Activation(activation, name=f"{prefix}_activation_2")(x)

    if se_ratio:
        x_se = layers.GlobalAveragePooling2D(
            keepdims=True, data_format=data_format, name=f"{prefix}_se_pool"
        )(x)
        x_se = layers.Conv2D(
            make_divisible(expanded_filters * se_ratio),
            kernel_size=1,
            padding="same",
            data_format=data_format,
            name=f"{prefix}_se_conv_1",
        )(x_se)
        x_se = layers.ReLU(name=f"{prefix}_se_activation_1")(x_se)
        x_se = layers.Conv2D(
            expanded_filters,
            kernel_size=1,
            padding="same",
            data_format=data_format,
            name=f"{prefix}_se_conv_2",
        )(x_se)
        x_se = layers.Activation("hard_sigmoid", name=f"{prefix}_se_activation_2")(x_se)
        x = layers.Multiply(name=f"{prefix}_se_multiply")([x, x_se])

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_conv_pwl",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=f"{prefix}_batchnorm_3",
    )(x)

    if stride == 1 and input_filters == filters:
        x = layers.Add(name=f"{prefix}_add")([shortcut, x])
    return x


_SMALL_BLOCKS = [
    # [expansion_ratio, filters, kernel_size, stride, se_ratio, activation]
    [1, 16, 3, 2, 0.25, "relu"],
    [72.0 / 16, 24, 3, 2, None, "relu"],
    [88.0 / 24, 24, 3, 1, None, "relu"],
    [4, 40, 5, 2, 0.25, "hard_swish"],
    [6, 40, 5, 1, 0.25, "hard_swish"],
    [6, 40, 5, 1, 0.25, "hard_swish"],
    [3, 48, 5, 1, 0.25, "hard_swish"],
    [3, 48, 5, 1, 0.25, "hard_swish"],
    [6, 96, 5, 2, 0.25, "hard_swish"],
    [6, 96, 5, 1, 0.25, "hard_swish"],
    [6, 96, 5, 1, 0.25, "hard_swish"],
]

_LARGE_BLOCKS = [
    [1, 16, 3, 1, None, "relu"],
    [4, 24, 3, 2, None, "relu"],
    [3, 24, 3, 1, None, "relu"],
    [3, 40, 5, 2, 0.25, "relu"],
    [3, 40, 5, 1, 0.25, "relu"],
    [3, 40, 5, 1, 0.25, "relu"],
    [6, 80, 3, 2, None, "hard_swish"],
    [2.5, 80, 3, 1, None, "hard_swish"],
    [2.3, 80, 3, 1, None, "hard_swish"],
    [2.3, 80, 3, 1, None, "hard_swish"],
    [6, 112, 3, 1, 0.25, "hard_swish"],
    [6, 112, 3, 1, 0.25, "hard_swish"],
    [6, 160, 5, 2, 0.25, "hard_swish"],
    [6, 160, 5, 1, 0.25, "hard_swish"],
    [6, 160, 5, 1, 0.25, "hard_swish"],
]


def _mobilenetv3_features(
    inputs,
    *,
    config,
    width_multiplier,
    depth_multiplier,
    minimal,
    data_format,
    channels_axis,
):
    """MobileNetV3 stem + IR stages + final 1x1 conv.

    Returns ``[stem, ir_0, ir_1, ..., ir_N, final_conv]``.
    """
    blocks = _LARGE_BLOCKS if config == "large" else _SMALL_BLOCKS
    features = []

    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        data_format=data_format,
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.999,
        name="stem_batchnorm",
    )(x)
    x = layers.Activation(
        "hard_swish" if not minimal else "relu", name="stem_activation"
    )(x)
    features.append(x)

    for idx, layer_config in enumerate(blocks):
        expansion_ratio, filters, kernel_size, stride, se_ratio, activation = (
            layer_config
        )
        if minimal:
            kernel_size = 3
            activation = "relu"
            se_ratio = None

        x = inverted_residual_block(
            x,
            expansion_ratio=expansion_ratio * depth_multiplier,
            filters=make_divisible(filters * width_multiplier),
            kernel_size=kernel_size,
            stride=stride,
            se_ratio=se_ratio,
            activation=activation,
            block_id=idx,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        features.append(x)

    final_conv_channels = make_divisible(x.shape[channels_axis] * 6)
    x = layers.Conv2D(
        final_conv_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name="final_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-3,
        momentum=0.999,
        name="final_batchnorm",
    )(x)
    x = layers.Activation(
        "hard_swish" if not minimal else "relu", name="final_activation"
    )(x)
    features.append(x)

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class MobileNetV3(BaseModel):
    """MobileNetV3 classifier (timm-ported).

    Reference:
    - [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) (ICCV 2019)

    Construction:

    >>> MobileNetV3.from_weights("mobilenetv3_large_100_ra_in1k")
    >>> MobileNetV3.from_weights("timm:timm/mobilenetv3_large_100.ra_in1k")
    """

    KMODELS_CONFIG = MOBILENETV3_CONFIG
    KMODELS_WEIGHTS = MOBILENETV3_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mobilenetv3_weights(keras_model, state_dict)

    def __init__(
        self,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        config="large",
        minimal=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        dropout_rate=0.2,
        name="MobileNetV3",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        if config not in ("large", "small"):
            raise ValueError(
                f"Invalid config. Expected 'large' or 'small', got {config!r}"
            )

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
        features = _mobilenetv3_features(
            x,
            config=config,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            minimal=minimal,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        head_channels = 1024 if config == "small" else 1280
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.Dense(
            head_channels,
            use_bias=True,
            name="head_conv",
        )(x)
        x = layers.Activation(
            "hard_swish" if not minimal else "relu", name="head_activation"
        )(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="head_dropout")(x)
        x = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.config = config
        self.minimal = minimal
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width_multiplier": self.width_multiplier,
                "depth_multiplier": self.depth_multiplier,
                "config": self.config,
                "minimal": self.minimal,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "dropout_rate": self.dropout_rate,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class MobileNetV3Backbone(BaseModel):
    """MobileNetV3 feature extractor (no classifier head)."""

    KMODELS_CONFIG = MOBILENETV3_CONFIG
    KMODELS_WEIGHTS = MOBILENETV3_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        return MobileNetV3

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
        transfer_mobilenetv3_weights(keras_model, state_dict)

    def __init__(
        self,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        config="large",
        minimal=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        name="MobileNetV3Backbone",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "dropout_rate", "timm_id"):
            kwargs.pop(k, None)

        if config not in ("large", "small"):
            raise ValueError(
                f"Invalid config. Expected 'large' or 'small', got {config!r}"
            )

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
        features = _mobilenetv3_features(
            x,
            config=config,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            minimal=minimal,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.config = config
        self.minimal = minimal
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width_multiplier": self.width_multiplier,
                "depth_multiplier": self.depth_multiplier,
                "config": self.config,
                "minimal": self.minimal,
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
