import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import MOBILENETV2_CONFIG, MOBILENETV2_WEIGHTS
from .convert_mobilenetv2_torch_to_keras import transfer_mobilenetv2_weights


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """
    Adjusts the given value `v` to be divisible by `divisor`,
        ensuring it meets the specified constraints.

    Args:
        v (int or float): The value to be adjusted.
        divisor (int, optional): The divisor to which `v` should be rounded. Default is 8.
        min_value (int, optional): The minimum allowed value. If None, it defaults to `divisor`.
        round_limit (float, optional): The threshold to increase `new_v` if it is too small.
            Default is 0.9.

    Returns:
        int: The adjusted value that is divisible by `divisor` and meets the
            given constraints.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def inverted_residual_block(
    x,
    filters,
    kernel_size,
    stride,
    expansion_ratio,
    channels_axis,
    data_format,
    block_id,
    sub_block_id,
):
    """MobileNetV2 inverted-residual block: expand 1x1, depthwise, project 1x1.

    Args:
        x: Input feature tensor.
        filters: Output channel count after the projection conv.
        kernel_size: Depthwise convolution kernel size.
        stride: Depthwise convolution stride.
        expansion_ratio: Expansion factor applied to the input channels.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        data_format: Keras data-format string.
        block_id: Stage index used to construct unique layer names.
        sub_block_id: Within-stage block index used to construct unique layer names.

    Returns:
        Output feature tensor with ``filters`` channels.
    """
    inputs = x
    block_name = f"blocks_{block_id}_{sub_block_id}"

    if expansion_ratio > 1:
        x = layers.Conv2D(
            make_divisible(x.shape[channels_axis] * expansion_ratio),
            1,
            1,
            use_bias=False,
            data_format=data_format,
            name=f"{block_name}_conv_pw",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name=f"{block_name}_batchnorm_1",
        )(x)
        x = layers.Activation("relu6", name=f"{block_name}_relu1")(x)

    if stride > 1:
        x = layers.ZeroPadding2D(
            padding=((1, 1), (1, 1)),
            data_format=data_format,
            name=f"{block_name}_zeropadding",
        )(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.DepthwiseConv2D(
        kernel_size,
        stride,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{block_name}_dwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{block_name}_batchnorm_2",
    )(x)
    x = layers.Activation("relu6", name=f"{block_name}_relu2")(x)

    x = layers.Conv2D(
        filters,
        1,
        1,
        use_bias=False,
        data_format=data_format,
        name=f"{block_name}_conv_pwl",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{block_name}_batchnorm_3",
    )(x)

    if stride == 1 and inputs.shape[channels_axis] == filters:
        x = layers.Add(name=f"{block_name}_add")([inputs, x])

    return x


_DEFAULT_BLOCKS = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


def mobilenetv2_backbone_feature(
    inputs,
    *,
    width_multiplier,
    depth_multiplier,
    fix_channels,
    data_format,
    channels_axis,
):
    """MobileNetV2 stem + 7 inverted-residual stages + head 1x1 conv.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        width_multiplier: Multiplier applied to per-stage channel counts.
        depth_multiplier: Multiplier applied to interior stage block repeats.
        fix_channels: If True, keep stem (32) and head (1280) channels fixed
            regardless of ``width_multiplier``.
        data_format: Keras data-format string.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).

    Returns:
        Final 4D feature tensor after the head 1x1 conv (post BN + ReLU6).
    """
    initial_dims = 32 if fix_channels else make_divisible(32 * width_multiplier)
    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), data_format=data_format, name="stem_padding"
    )(inputs)
    x = layers.Conv2D(
        initial_dims,
        3,
        2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name="stem_batchnorm",
    )(x)
    x = layers.Activation("relu6", name="relu1")(x)

    for layer_idx, layer_config in enumerate(_DEFAULT_BLOCKS):
        expansion_factor, output_channels, num_blocks, initial_stride = layer_config
        scaled_output_channels = make_divisible(output_channels * width_multiplier)

        if layer_idx not in (0, len(_DEFAULT_BLOCKS) - 1):
            num_blocks = int(keras.ops.ceil(num_blocks * depth_multiplier))

        for block_idx in range(num_blocks):
            current_stride = initial_stride if block_idx == 0 else 1
            x = inverted_residual_block(
                x,
                filters=scaled_output_channels,
                kernel_size=3,
                stride=current_stride,
                expansion_ratio=expansion_factor,
                channels_axis=channels_axis,
                data_format=data_format,
                block_id=layer_idx,
                sub_block_id=block_idx,
            )

    head_dims = (
        1280
        if fix_channels or width_multiplier <= 1.0
        else make_divisible(1280 * width_multiplier)
    )
    x = layers.Conv2D(
        head_dims,
        1,
        1,
        use_bias=False,
        data_format=data_format,
        name="head_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name="head_batchnorm",
    )(x)
    x = layers.Activation("relu6", name="relu2")(x)

    return x


@keras.saving.register_keras_serializable(package="kmodels")
class MobileNetV2Model(BaseModel):
    """MobileNetV2 backbone — returns the post-head-conv 4D feature map.

    Output shape: ``(B, H, W, C)`` — head-conv 4D feature map after the final
    1x1 conv + BN + ReLU6. :class:`MobileNetV2Classify` composes this model
    and adds GlobalAveragePool + Dense on top.
    """

    KMODELS_CONFIG = MOBILENETV2_CONFIG
    KMODELS_WEIGHTS = MOBILENETV2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = MobileNetV2Classify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mobilenetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        fix_channels=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="MobileNetV2Model",
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
        x = mobilenetv2_backbone_feature(
            x,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            fix_channels=fix_channels,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.fix_channels = fix_channels
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
                "fix_channels": self.fix_channels,
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
class MobileNetV2Classify(BaseModel):
    """MobileNetV2 classifier (timm-ported).

    Wraps a :class:`MobileNetV2Model` backbone and adds GlobalAveragePool +
    Dense on top.

    Reference:
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
        https://arxiv.org/abs/1801.04381) (CVPR 2018)

    Construction:

    >>> MobileNetV2Classify.from_weights("mobilenetv2_100_ra_in1k")
    >>> MobileNetV2Classify.from_weights("timm:timm/mobilenetv2_100.ra_in1k")
    """

    KMODELS_CONFIG = MOBILENETV2_CONFIG
    KMODELS_WEIGHTS = MOBILENETV2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mobilenetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        fix_channels=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="MobileNetV2Classify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = MobileNetV2Model(
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
            fix_channels=fix_channels,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        out = layers.Dense(
            num_classes,
            use_bias=True,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.fix_channels = fix_channels
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
                "width_multiplier": self.width_multiplier,
                "depth_multiplier": self.depth_multiplier,
                "fix_channels": self.fix_channels,
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
