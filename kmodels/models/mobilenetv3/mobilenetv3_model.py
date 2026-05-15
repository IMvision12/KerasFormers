import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import MOBILENETV3_MODEL_CONFIG, MOBILENETV3_WEIGHT_CONFIG
from .convert_mobilenetv3_keras_to_keras import transfer_mobilenetv3_weights


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """Snap a (possibly scaled) channel count to a multiple of ``divisor``.

    Args:
        v: Value to be adjusted.
        divisor: Multiple to which ``v`` should be rounded.
        min_value: Minimum allowed value (defaults to ``divisor``).
        round_limit: Lower-bound ratio that triggers bumping the result up by
            one ``divisor`` when rounding-down went too far.

    Returns:
        Adjusted value divisible by ``divisor`` and at least ``min_value``.
    """
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
    """MobileNetV3-style inverted residual block with optional Squeeze-and-Excitation.

    Args:
        x: Input feature tensor.
        expansion_ratio: Expansion factor applied to the input channel count.
        filters: Output channel count after the projection conv.
        kernel_size: Depthwise convolution kernel size.
        stride: Depthwise convolution stride.
        se_ratio: Squeeze-and-Excitation ratio (or ``None``/0 to disable).
        activation: Activation name used after the expand and depthwise convs.
        block_id: Block index used to construct unique layer names.
        data_format: Keras data-format string.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).

    Returns:
        Output feature tensor with ``filters`` channels.
    """
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


def mobilenetv3_backbone_feature(
    inputs,
    *,
    config,
    width_multiplier,
    depth_multiplier,
    minimal,
    data_format,
    channels_axis,
    return_stages=False,
):
    """MobileNetV3 stem + inverted-residual stages + final 1x1 conv.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        config: Variant key, ``"large"`` or ``"small"``, selecting the block table.
        width_multiplier: Multiplier applied to per-stage channel counts.
        depth_multiplier: Multiplier applied to per-block expansion ratios.
        minimal: If True, force kernel size 3, ReLU activations, and disable SE
            for every IR block (minimal variant).
        data_format: Keras data-format string.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        return_stages: If True, return a list of per-stage feature maps grouped
            by stride boundary (pre-final-conv); otherwise return the
            post-final-conv tensor.

    Returns:
        Final 4D feature tensor after the final 1x1 conv (post BN + activation),
        or a list of per-stage feature tensors when ``return_stages`` is True.
    """
    blocks = _LARGE_BLOCKS if config == "large" else _SMALL_BLOCKS

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

    stages = []
    for idx, layer_config in enumerate(blocks):
        expansion_ratio, filters, kernel_size, stride, se_ratio, activation = (
            layer_config
        )
        if minimal:
            kernel_size = 3
            activation = "relu"
            se_ratio = None

        if return_stages and stride == 2:
            stages.append(x)

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

    if return_stages:
        stages.append(x)
        return stages

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

    return x


@keras.saving.register_keras_serializable(package="kmodels")
class MobileNetV3Model(BaseModel):
    """MobileNetV3 backbone — returns the post-final-conv 4D feature map.

    Output shape: ``(B, H, W, C)`` — feature map after the final 1x1 conv +
    BN + activation. :class:`MobileNetV3Classify` composes this model and
    adds GlobalAveragePool + Dense(head_channels) + activation + Dropout +
    Dense(num_classes) on top.
    """

    KMODELS_CONFIG = MOBILENETV3_MODEL_CONFIG
    KMODELS_WEIGHTS = MOBILENETV3_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = MobileNetV3Classify.from_weights(variant)
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
        as_backbone=False,
        name="MobileNetV3Model",
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
        x = mobilenetv3_backbone_feature(
            x,
            config=config,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            minimal=minimal,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.config = config
        self.minimal = minimal
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

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
                "as_backbone": self.as_backbone,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class MobileNetV3Classify(BaseModel):
    """MobileNetV3 classifier (timm-ported).

    Wraps a :class:`MobileNetV3Model` backbone and adds GlobalAveragePool +
    Dense(head_channels) + activation + (optional Dropout) + Dense(num_classes)
    on top.

    Reference:
    - [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) (ICCV 2019)

    Construction:

    >>> MobileNetV3Classify.from_weights("mobilenetv3_large_100_ra_in1k")
    >>> MobileNetV3Classify.from_weights("timm:timm/mobilenetv3_large_100.ra_in1k")
    """

    KMODELS_CONFIG = MOBILENETV3_MODEL_CONFIG
    KMODELS_WEIGHTS = MOBILENETV3_WEIGHT_CONFIG
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
        name="MobileNetV3Classify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        if config not in ("large", "small"):
            raise ValueError(
                f"Invalid config. Expected 'large' or 'small', got {config!r}"
            )

        data_format = keras.config.image_data_format()

        backbone = MobileNetV3Model(
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            config=config,
            minimal=minimal,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        head_channels = 1024 if config == "small" else 1280
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
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
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

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
