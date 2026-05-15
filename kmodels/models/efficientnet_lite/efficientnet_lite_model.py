import copy
import math

import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import (
    CONV_KERNEL_INITIALIZER,
    DEFAULT_BLOCKS_ARGS,
    DENSE_KERNEL_INITIALIZER,
    EFFICIENTNET_LITE_CONFIG,
    EFFICIENTNET_LITE_WEIGHTS,
)
from .convert_efficientnet_lite_torch_to_keras import transfer_efficientnet_lite_weights


def round_filters(filters, width_coefficient, divisor=8):
    """Round filter count by ``width_coefficient`` and snap to a multiple of ``divisor``.

    Args:
        filters: Base filter count to scale.
        width_coefficient: Multiplier applied to ``filters`` before rounding.
        divisor: Multiple to which the rounded count is snapped.

    Returns:
        Adjusted integer filter count satisfying the divisibility constraint.
    """
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round-up repeat count by ``depth_coefficient``.

    Args:
        repeats: Base number of block repeats.
        depth_coefficient: Depth multiplier applied to ``repeats``.

    Returns:
        Integer repeat count after ceiling.
    """
    return int(math.ceil(depth_coefficient * repeats))


def efficientnetlite_block(
    inputs,
    channels_axis,
    data_format,
    drop_rate=0.0,
    name="",
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
    id_skip=True,
):
    """MBConv-Lite block (no SE, ReLU6 activations).

    Args:
        inputs: Input feature tensor.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        data_format: Keras data-format string.
        drop_rate: Dropout rate applied to the residual branch when the skip is active.
        name: Prefix used to name the layers inside the block.
        filters_in: Number of input channels.
        filters_out: Number of output channels.
        kernel_size: Depthwise convolution kernel size.
        strides: Depthwise convolution stride.
        expand_ratio: Expansion factor for the inverted residual.
        id_skip: Whether to add the identity skip connection (when shapes match).

    Returns:
        Output feature tensor with ``filters_out`` channels.
    """
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=name + "conv2d_1",
        )(inputs)
        x = layers.BatchNormalization(axis=channels_axis, name=name + "batchnorm_1")(x)
        x = layers.ReLU(max_value=6, name=name + "activation1")(x)
    else:
        x = inputs

    if strides == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            name=name + "dwconv_pad",
            data_format=data_format,
        )(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        data_format=data_format,
        name=name + "dwconv2d",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, name=name + "batchnorm_2")(x)
    x = layers.ReLU(max_value=6, name=name + "activation2")(x)

    x = layers.Conv2D(
        filters_out,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        data_format=data_format,
        name=name + "conv2d_2",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, name=name + "batchnorm_3")(x)
    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)
        x = layers.add([x, inputs], name=name + "add")
    return x


def efficientnet_lite_backbone_feature(
    inputs,
    *,
    width_coefficient,
    depth_coefficient,
    drop_connect_rate,
    data_format,
    channels_axis,
    return_stages=False,
):
    """EfficientNet-Lite stem + MBConv-Lite stages + head conv.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        width_coefficient: Filter-count multiplier (stem/head channels are kept fixed).
        depth_coefficient: Depth multiplier applied to interior stage repeats only.
        drop_connect_rate: Stochastic-depth drop rate ramp applied across blocks.
        data_format: Keras data-format string.
        channels_axis: Channel axis (``-1`` for channels-last, ``1`` for channels-first).
        return_stages: If True, return a list of per-stage feature maps grouped
            by stride boundary (pre-head-conv); otherwise return the post-head-conv
            tensor.

    Returns:
        Final 4D feature tensor after the head 1x1 conv (post BN + ReLU6), or a
        list of per-stage feature tensors when ``return_stages`` is True.
    """
    x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(inputs, 3),
        data_format=data_format,
        name="stem_conv_pad",
    )(inputs)
    x = layers.Conv2D(
        32,
        3,
        strides=2,
        padding="valid",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        data_format=data_format,
        name="conv_stem",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, name="batchnorm_1")(x)
    x = layers.ReLU(max_value=6, name="stem_activation")(x)

    blocks_args = copy.deepcopy(DEFAULT_BLOCKS_ARGS)
    b = 0
    blocks = float(sum(args["repeats"] for args in DEFAULT_BLOCKS_ARGS))

    stages = []
    for i, args in enumerate(blocks_args):
        args["filters_in"] = round_filters(args["filters_in"], width_coefficient)
        args["filters_out"] = round_filters(args["filters_out"], width_coefficient)
        if i == 0 or i == (len(blocks_args) - 1):
            repeats = args.pop("repeats")
        else:
            repeats = round_repeats(args.pop("repeats"), depth_coefficient)

        group_stride = args["strides"]
        if return_stages and group_stride == 2:
            stages.append(x)

        for j in range(repeats):
            if j > 0:
                args["strides"] = 1
                args["filters_in"] = args["filters_out"]
            x = efficientnetlite_block(
                x,
                channels_axis,
                data_format,
                drop_connect_rate * b / blocks,
                name=f"blocks_{i}_{j}_",
                **args,
            )
            b += 1

    if return_stages:
        stages.append(x)
        return stages

    x = layers.Conv2D(
        1280,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        data_format=data_format,
        name="conv_head",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, name="batchnorm_2")(x)
    x = layers.ReLU(max_value=6, name="top_activation")(x)
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class EfficientNetLiteModel(BaseModel):
    """EfficientNet-Lite backbone — returns the post-head-conv 4D feature map.

    Output shape: ``(B, H, W, C)`` — head-conv 4D feature map after the final
    1x1 conv + BN + ReLU6. :class:`EfficientNetLiteClassify` composes this
    model and adds GlobalAveragePool + Dropout + Dense on top.
    """

    KMODELS_CONFIG = EFFICIENTNET_LITE_CONFIG
    KMODELS_WEIGHTS = EFFICIENTNET_LITE_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = EfficientNetLiteClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_efficientnet_lite_weights(keras_model, state_dict)

    def __init__(
        self,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        as_backbone=False,
        name="EfficientNetLiteModel",
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
        x = efficientnet_lite_backbone_feature(
            x,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            drop_connect_rate=drop_connect_rate,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "default_size": self.default_size,
                "dropout_rate": self.dropout_rate,
                "drop_connect_rate": self.drop_connect_rate,
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
class EfficientNetLiteClassify(BaseModel):
    """EfficientNet-Lite classifier (timm-ported).

    Wraps a :class:`EfficientNetLiteModel` backbone and adds GlobalAveragePool
    + Dropout + Dense on top.

    Construction:

    >>> EfficientNetLiteClassify.from_weights("tf_efficientnet_lite0_in1k")
    >>> EfficientNetLiteClassify.from_weights("timm:timm/tf_efficientnet_lite0.in1k")
    """

    KMODELS_CONFIG = EFFICIENTNET_LITE_CONFIG
    KMODELS_WEIGHTS = EFFICIENTNET_LITE_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_efficientnet_lite_weights(keras_model, state_dict)

    def __init__(
        self,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        num_classes=1000,
        classifier_activation="linear",
        name="EfficientNetLiteClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = EfficientNetLiteModel(
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            default_size=default_size,
            dropout_rate=dropout_rate,
            drop_connect_rate=drop_connect_rate,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            input_shape=input_shape,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="dropout")(x)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
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
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "default_size": self.default_size,
                "dropout_rate": self.dropout_rate,
                "drop_connect_rate": self.drop_connect_rate,
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
