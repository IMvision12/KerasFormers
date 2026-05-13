"""EfficientNetV2 classifier and backbone (timm-ported)."""

import copy
import math

import keras
from keras import initializers, layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import (
    CONV_KERNEL_INITIALIZER,
    DENSE_KERNEL_INITIALIZER,
    EFFICIENTNETV2_BLOCK_CONFIG,
    EFFICIENTNETV2_CONFIG,
    EFFICIENTNETV2_WEIGHTS,
)
from .convert_efficientnetv2_torch_to_keras import transfer_efficientnetv2_weights


def round_filters(filters, width_coefficient, min_depth=8, depth_divisor=8):
    """Round filter count by ``width_coefficient`` and snap to a multiple of ``depth_divisor``."""
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round-up repeat count by ``depth_coefficient``."""
    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(
    inputs,
    input_filters,
    output_filters,
    channels_axis,
    data_format,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    survival_probability=0.8,
    block_idx=0,
    layer_idx=0,
):
    """Mobile Inverted Residual Block (MBConv) with optional SE and stochastic depth."""
    block_name = f"blocks_{block_idx}_{layer_idx}_"

    filters = input_filters * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=block_name + "MBconv1",
        )(inputs)
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            name=block_name + "batchnorm1",
        )(x)
        x = layers.Activation("swish", name=block_name + "act1")(x)
    else:
        x = inputs

    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=block_name + "MBdwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, name=block_name + "batchnorm2"
    )(x)
    x = layers.Activation("swish", name=block_name + "act2")(x)

    if 0 < se_ratio <= 1:
        filters_se = max(1, int(input_filters * se_ratio))
        se = layers.GlobalAveragePooling2D(
            data_format=data_format, name=block_name + "se_avgpool"
        )(x)
        if channels_axis == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)
        se = layers.Reshape(se_shape, name=block_name + "se_reshape")(se)

        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation="swish",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=block_name + "se_conv_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=block_name + "se_conv_expand",
        )(se)

        x = layers.multiply([x, se], name=block_name + "se_excite")

    x = layers.Conv2D(
        filters=output_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=block_name + "MBconv2",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, name=block_name + "batchnorm3"
    )(x)

    if strides == 1 and input_filters == output_filters:
        if survival_probability:
            x = layers.Dropout(
                survival_probability,
                noise_shape=(None, 1, 1, 1),
                name=block_name + "dropout",
            )(x)
        x = layers.add([x, inputs], name=block_name + "add")

    return x


def fusedmb_conv_block(
    inputs,
    input_filters,
    output_filters,
    channels_axis,
    data_format,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    survival_probability=0.8,
    block_idx=0,
    layer_idx=0,
):
    """Fused Mobile Inverted Residual Block (FusedMBConv) with optional SE and stochastic depth."""
    block_name = f"blocks_{block_idx}_{layer_idx}_"

    filters = input_filters * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=block_name + "FMBconv1",
        )(inputs)
        x = layers.BatchNormalization(
            axis=channels_axis, momentum=0.9, name=block_name + "batchnorm1"
        )(x)
        x = layers.Activation(activation="swish", name=block_name + "act1")(x)
    else:
        x = inputs

    if 0 < se_ratio <= 1:
        filters_se = max(1, int(input_filters * se_ratio))
        se = layers.GlobalAveragePooling2D(
            data_format=data_format, name=block_name + "se_avgpool"
        )(x)
        if channels_axis == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)

        se = layers.Reshape(se_shape, name=block_name + "se_reshape")(se)

        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation="swish",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=block_name + "se_conv_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            data_format=data_format,
            name=block_name + "se_conv_expand",
        )(se)

        x = layers.multiply([x, se], name=block_name + "se_excite")

    x = layers.Conv2D(
        output_filters,
        kernel_size=1 if expand_ratio != 1 else kernel_size,
        strides=1 if expand_ratio != 1 else strides,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=block_name + "FMBconv2",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, name=block_name + "batchnorm2"
    )(x)
    if expand_ratio == 1:
        x = layers.Activation(activation="swish", name=block_name + "act2")(x)

    if strides == 1 and input_filters == output_filters:
        if survival_probability:
            x = layers.Dropout(
                survival_probability,
                noise_shape=(None, 1, 1, 1),
                name=block_name + "dropout",
            )(x)
        x = layers.add([x, inputs], name=block_name + "add")

    return x


def _efficientnetv2_features(
    inputs,
    *,
    width_coefficient,
    depth_coefficient,
    block_arch,
    head_filters,
    data_format,
    channels_axis,
):
    """EfficientNetV2 stem + 6/7 stages + head conv, returns ``[stem, s1..sN, head]``."""
    features = []

    block_config = copy.deepcopy(EFFICIENTNETV2_BLOCK_CONFIG[block_arch])

    stem_filters = round_filters(
        filters=block_config[0]["input_filters"],
        width_coefficient=width_coefficient,
    )
    x = layers.Conv2D(
        filters=stem_filters,
        kernel_size=3,
        strides=2,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name="conv_stem",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        name="batchnorm1",
    )(x)
    x = layers.Activation("swish", name="act1")(x)
    features.append(x)

    b = 0
    blocks = float(sum(args["num_repeat"] for args in block_config))

    for i, args in enumerate(block_config):
        assert args["num_repeat"] > 0

        args["input_filters"] = round_filters(
            filters=args["input_filters"],
            width_coefficient=width_coefficient,
        )
        args["output_filters"] = round_filters(
            filters=args["output_filters"],
            width_coefficient=width_coefficient,
        )

        block = {0: mb_conv_block, 1: fusedmb_conv_block}[args.pop("conv_type")]
        repeats = round_repeats(
            repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient
        )
        for j in range(repeats):
            if j > 0:
                args["strides"] = 1
                args["input_filters"] = args["output_filters"]

            x = block(
                x,
                survival_probability=0.2 * b / blocks,
                block_idx=i,
                layer_idx=j,
                data_format=data_format,
                channels_axis=channels_axis,
                **args,
            )
            b += 1

        features.append(x)

    x = layers.Conv2D(
        filters=head_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        use_bias=False,
        padding="same",
        data_format=data_format,
        name="conv_head",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        name="batchnorm2",
    )(x)
    x = layers.Activation(activation="swish", name="act2")(x)
    features.append(x)

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class EfficientNetV2Classify(BaseModel):
    """EfficientNetV2 classifier (timm-ported).

    Reference:
    - [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)

    Construction:

    >>> EfficientNetV2Classify.from_weights("tf_efficientnetv2_s_in21k_ft_in1k")
    >>> EfficientNetV2Classify.from_weights("timm:timm/tf_efficientnetv2_s.in21k_ft_in1k")
    """

    KMODELS_CONFIG = EFFICIENTNETV2_CONFIG
    KMODELS_WEIGHTS = EFFICIENTNETV2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_efficientnetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=300,
        block_arch="EfficientNetV2S",
        head_filters=1280,
        image_size=300,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="EfficientNetV2Classify",
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
        features = _efficientnetv2_features(
            x,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            block_arch=block_arch,
            head_filters=head_filters,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.Dropout(0.2, name="top_dropout")(x)
        x = layers.Dense(
            num_classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            bias_initializer=initializers.Constant(0.0),
            name="predictions",
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.block_arch = block_arch
        self.head_filters = head_filters
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
                "block_arch": self.block_arch,
                "head_filters": self.head_filters,
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
class EfficientNetV2Backbone(BaseModel):
    """EfficientNetV2 feature extractor. Returns ``[stem, s1..sN, head_conv]``."""

    KMODELS_CONFIG = EFFICIENTNETV2_CONFIG
    KMODELS_WEIGHTS = EFFICIENTNETV2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        return EfficientNetV2Classify

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
        transfer_efficientnetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=300,
        block_arch="EfficientNetV2S",
        head_filters=1280,
        image_size=300,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        name="EfficientNetV2Backbone",
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
        features = _efficientnetv2_features(
            x,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            block_arch=block_arch,
            head_filters=head_filters,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.block_arch = block_arch
        self.head_filters = head_filters
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "default_size": self.default_size,
                "block_arch": self.block_arch,
                "head_filters": self.head_filters,
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
class EfficientNetV2Model(BaseModel):
    """EfficientNetV2 trunk returning the final stage feature map."""

    KMODELS_CONFIG = EFFICIENTNETV2_CONFIG
    KMODELS_WEIGHTS = EFFICIENTNETV2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        return EfficientNetV2Classify

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
        transfer_efficientnetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=300,
        block_arch="EfficientNetV2S",
        head_filters=1280,
        image_size=300,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        name="EfficientNetV2Model",
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
        features = _efficientnetv2_features(
            x,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            block_arch=block_arch,
            head_filters=head_filters,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features[-1], name=name, **kwargs)

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.block_arch = block_arch
        self.head_filters = head_filters
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "default_size": self.default_size,
                "block_arch": self.block_arch,
                "head_filters": self.head_filters,
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
