import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils
from keras.src.utils.argument_validation import standardize_tuple

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import INCEPTION_RESNET_V2_CONFIG, INCEPTION_RESNET_V2_WEIGHTS
from .convert_inceptionresnetv2_torch_to_keras import (
    transfer_inception_resnet_v2_weights,
)


def conv_block(
    inputs,
    filters=None,
    kernel_size=1,
    strides=1,
    bn_momentum=0.9,
    bn_epsilon=1e-3,
    padding="valid",
    name="conv2d_block",
):
    """Conv -> BatchNorm -> ReLU with optional asymmetric ZeroPadding."""
    kernel_size = standardize_tuple(kernel_size, 2, "kernel_size")
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    x = inputs

    if padding is None:

        def calculate_padding(kernel_dim):
            pad_total = kernel_dim - 1
            pad_size = pad_total // 2
            pad_extra = (kernel_dim - 1) % 2
            return pad_size, pad_extra

        pad_h, extra_h = calculate_padding(kernel_size[0])
        pad_w, extra_w = calculate_padding(kernel_size[1])

        if strides > 1:
            padding_config = ((pad_h + extra_h, pad_h), (pad_w + extra_w, pad_w))
        else:
            padding_config = ((pad_h, pad_h), (pad_w, pad_w))

        x = layers.ZeroPadding2D(padding=padding_config, name=f"{name}_padding")(x)
        padding = "valid"

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=keras.config.image_data_format(),
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        name=f"{name}_batchnorm",
    )(x)
    x = layers.Activation("relu", name=name)(x)
    return x


def mixed_5b_block(inputs, name="mixed_5b"):
    """Stem-end Mixed-5b block (1x1, 5x5, double-3x3, avg-pool)."""
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 96, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 48, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 64, 5, padding="same", name=f"{name}_branch1_1")

    branch2 = conv_block(inputs, 64, 1, name=f"{name}_branch2_0")
    branch2 = conv_block(branch2, 96, 3, padding="same", name=f"{name}_branch2_1")
    branch2 = conv_block(branch2, 96, 3, padding="same", name=f"{name}_branch2_2")

    branch_pool = layers.AveragePooling2D(
        pool_size=3,
        strides=1,
        padding="same",
        data_format=keras.config.image_data_format(),
    )(inputs)
    branch_pool = conv_block(branch_pool, 64, name=f"{name}_branch3_1")

    return layers.Concatenate(axis=channels_axis)(
        [branch0, branch1, branch2, branch_pool]
    )


def block35(inputs, scale=1.0, name="repeat_0"):
    """Inception-ResNet-A residual block."""
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 32, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 32, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 32, 3, padding="same", name=f"{name}_branch1_1")

    branch2 = conv_block(inputs, 32, 1, name=f"{name}_branch2_0")
    branch2 = conv_block(branch2, 48, 3, padding="same", name=f"{name}_branch2_1")
    branch2 = conv_block(branch2, 64, 3, padding="same", name=f"{name}_branch2_2")

    branches = [branch0, branch1, branch2]
    mixed = layers.Concatenate(axis=channels_axis)(branches)
    up = layers.Conv2D(320, 1, use_bias=True, name=f"{name}_conv2d")(mixed)

    x = layers.Lambda(lambda inputs: inputs[0] + inputs[1] * scale)([inputs, up])
    x = layers.Activation("relu", name=name)(x)
    return x


def mixed_6a_block(inputs, name="mixed_6a"):
    """Reduction-A block."""
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(
        inputs, 384, 3, strides=2, padding="valid", name=f"{name}_branch0"
    )

    branch1 = conv_block(inputs, 256, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 256, 3, padding="same", name=f"{name}_branch1_1")
    branch1 = conv_block(
        branch1, 384, 3, strides=2, padding="valid", name=f"{name}_branch1_2"
    )

    branch_pool = layers.MaxPooling2D(pool_size=3, strides=2)(inputs)

    return layers.Concatenate(axis=channels_axis)([branch0, branch1, branch_pool])


def block17(inputs, scale=1.0, name="repeat_1_0"):
    """Inception-ResNet-B residual block."""
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 192, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 128, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 160, (1, 7), padding="same", name=f"{name}_branch1_1")
    branch1 = conv_block(branch1, 192, (7, 1), padding="same", name=f"{name}_branch1_2")

    branches = [branch0, branch1]
    mixed = layers.Concatenate(axis=channels_axis)(branches)
    up = layers.Conv2D(1088, 1, use_bias=True, name=f"{name}_conv2d")(mixed)

    x = layers.Lambda(lambda inputs: inputs[0] + inputs[1] * scale)([inputs, up])
    x = layers.Activation("relu", name=name)(x)
    return x


def mixed_7a_block(inputs, name="mixed_7a"):
    """Reduction-B block."""
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 256, 1, name=f"{name}_branch0_0")
    branch0 = conv_block(
        branch0, 384, 3, strides=2, padding="valid", name=f"{name}_branch0_1"
    )

    branch1 = conv_block(inputs, 256, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1, 288, 3, strides=2, padding="valid", name=f"{name}_branch1_1"
    )

    branch2 = conv_block(inputs, 256, 1, name=f"{name}_branch2_0")
    branch2 = conv_block(branch2, 288, 3, padding="same", name=f"{name}_branch2_1")
    branch2 = conv_block(
        branch2, 320, 3, strides=2, padding="valid", name=f"{name}_branch2_2"
    )

    branch_pool = layers.MaxPooling2D(pool_size=3, strides=2)(inputs)

    return layers.Concatenate(axis=channels_axis)(
        [branch0, branch1, branch2, branch_pool]
    )


def block8(inputs, scale=1.0, activation=True, name="repeat_2_0"):
    """Inception-ResNet-C residual block."""
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch0 = conv_block(inputs, 192, 1, name=f"{name}_branch0")

    branch1 = conv_block(inputs, 192, 1, name=f"{name}_branch1_0")
    branch1 = conv_block(branch1, 224, (1, 3), padding="same", name=f"{name}_branch1_1")
    branch1 = conv_block(branch1, 256, (3, 1), padding="same", name=f"{name}_branch1_2")

    branches = [branch0, branch1]
    mixed = layers.Concatenate(axis=channels_axis)(branches)
    up = layers.Conv2D(2080, 1, use_bias=True, name=f"{name}_conv2d")(mixed)

    x = layers.Lambda(lambda inputs: inputs[0] + inputs[1] * scale)([inputs, up])
    if activation:
        x = layers.Activation("relu", name=name)(x)
    return x


def inception_resnet_v2_backbone_feature(inputs, *, data_format):
    """InceptionResNetV2 full backbone, returns a list of stage feature maps."""
    features = []

    x = conv_block(inputs, 32, 3, strides=2, padding="valid", name="conv2d_1a")
    x = conv_block(x, 32, 3, padding="valid", name="conv2d_2a")
    x = conv_block(x, 64, 3, padding="same", name="conv2d_2b")
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv_block(x, 80, 1, name="conv2d_3b")
    x = conv_block(x, 192, 3, padding="valid", name="conv2d_4a")
    x = layers.MaxPooling2D(3, strides=2)(x)
    features.append(x)

    x = mixed_5b_block(x, name="mixed_5b")
    for i in range(10):
        x = block35(x, scale=0.17, name=f"repeat_{i}")
    features.append(x)

    x = mixed_6a_block(x, name="mixed_6a")
    for i in range(20):
        x = block17(x, scale=0.10, name=f"repeats_1_{i}")
    features.append(x)

    x = mixed_7a_block(x, name="mixed_7a")
    for i in range(9):
        x = block8(x, scale=0.20, name=f"repeats_2_{i}")
    x = block8(x, activation=False, name="block8")
    x = conv_block(x, 1536, 1, name="conv2d_7b")
    features.append(x)

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class InceptionResNetV2Classify(BaseModel):
    """Inception-ResNet-v2 classifier (timm-ported).

    Reference:
    - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) (AAAI 2017)

    Construction:

    >>> InceptionResNetV2Classify.from_weights("inception_resnet_v2_tf_in1k")
    >>> InceptionResNetV2Classify.from_weights("timm:timm/inception_resnet_v2.tf_in1k")
    """

    KMODELS_CONFIG = INCEPTION_RESNET_V2_CONFIG
    KMODELS_WEIGHTS = INCEPTION_RESNET_V2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_inception_resnet_v2_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="InceptionResNetV2Classify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=image_size,
            min_size=75,
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
        features = inception_resnet_v2_backbone_feature(x, data_format=data_format)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(features[-1])
        x = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
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
class InceptionResNetV2Backbone(BaseModel):
    """InceptionResNetV2 feature extractor (4 stage maps)."""

    KMODELS_CONFIG = INCEPTION_RESNET_V2_CONFIG
    KMODELS_WEIGHTS = INCEPTION_RESNET_V2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = InceptionResNetV2Classify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_inception_resnet_v2_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        name="InceptionResNetV2Backbone",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=image_size,
            min_size=75,
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
        features = inception_resnet_v2_backbone_feature(x, data_format=data_format)

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
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
class InceptionResNetV2Model(BaseModel):
    """InceptionResNetV2 trunk returning the final stage feature map.

    Output shape: ``(B, H, W, C)`` — unpooled, head-free.
    """

    KMODELS_CONFIG = INCEPTION_RESNET_V2_CONFIG
    KMODELS_WEIGHTS = INCEPTION_RESNET_V2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = InceptionResNetV2Classify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_inception_resnet_v2_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        name="InceptionResNetV2Model",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=image_size,
            min_size=75,
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
        features = inception_resnet_v2_backbone_feature(x, data_format=data_format)

        super().__init__(inputs=img_input, outputs=features[-1], name=name, **kwargs)

        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
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
