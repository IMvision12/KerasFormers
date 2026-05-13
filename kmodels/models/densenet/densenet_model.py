import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import DENSENET_CONFIG, DENSENET_WEIGHTS
from .convert_densenet_torch_to_keras import transfer_densenet_weights


def conv_block(
    x,
    growth_rate,
    expansion_ratio,
    channels_axis,
    data_format,
    name,
):
    """Single conv layer inside a DenseNet dense block."""
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_1"
    )(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(
        int(growth_rate * expansion_ratio),
        kernel_size=1,
        strides=1,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv2d_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_2"
    )(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = layers.Conv2D(
        growth_rate,
        3,
        1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv2d_2",
    )(x)
    return x


def densenet_block(
    x,
    num_layers,
    growth_rate,
    channels_axis,
    data_format,
    name,
):
    """Dense block: stack of conv blocks with channel-wise concatenation."""
    output = x

    for i in range(num_layers):
        layer_output = conv_block(
            output,
            growth_rate,
            expansion_ratio=4.0,
            channels_axis=channels_axis,
            data_format=data_format,
            name=f"{name}_denselayer{i + 1}",
        )
        output = layers.Concatenate(axis=channels_axis)([output, layer_output])

    return output


def transition_block(x, reduction, channels_axis, data_format, name):
    """Reduce channels by ``reduction`` and 2x downsample spatially."""
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_transition_batchnorm",
    )(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    x = layers.Conv2D(
        int(x.shape[channels_axis] * reduction),
        1,
        1,
        "same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_transition_conv2d",
    )(x)
    x = layers.AveragePooling2D(
        2, 2, data_format=data_format, name=f"{name}_transition_pool"
    )(x)
    return x


def _densenet_features(
    inputs,
    *,
    num_blocks,
    growth_rate,
    initial_filter,
    channels_axis,
    data_format,
):
    """Stem + N dense blocks + final BN, returning ``[stem, block1..blockN]``."""
    features = []

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), data_format=data_format)(inputs)
    x = layers.Conv2D(
        initial_filter,
        7,
        2,
        use_bias=False,
        data_format=data_format,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_norm"
    )(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.ZeroPadding2D(1, data_format=data_format)(x)
    x = layers.MaxPooling2D(3, 2, data_format=data_format, name="stem_pool")(x)
    features.append(x)

    for i, num_layers in enumerate(num_blocks):
        x = densenet_block(
            x,
            num_layers,
            growth_rate,
            channels_axis,
            data_format,
            name=f"dense_block{i + 1}",
        )

        if i != len(num_blocks) - 1:
            x = transition_block(
                x,
                0.5,
                channels_axis,
                data_format,
                name=f"transition_block{i + 1}",
            )
        features.append(x)

    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name="final_batchnorm",
    )(x)
    x = layers.ReLU(name="final_relu")(x)
    features[-1] = x

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class DenseNet(BaseModel):
    """DenseNet classifier (timm-ported).

    Reference:
    - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

    Construction:

    >>> DenseNet.from_weights("densenet121_tv_in1k")
    >>> DenseNet.from_weights("timm:timm/densenet121.tv_in1k")
    """

    KMODELS_CONFIG = DENSENET_CONFIG
    KMODELS_WEIGHTS = DENSENET_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_densenet_weights(keras_model, state_dict)

    def __init__(
        self,
        num_blocks=(6, 12, 24, 16),
        growth_rate=32,
        initial_filter=64,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="DenseNet",
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
        features = _densenet_features(
            x,
            num_blocks=num_blocks,
            growth_rate=growth_rate,
            initial_filter=initial_filter,
            channels_axis=channels_axis,
            data_format=data_format,
        )
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.num_blocks = num_blocks
        self.growth_rate = growth_rate
        self.initial_filter = initial_filter
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
                "num_blocks": self.num_blocks,
                "growth_rate": self.growth_rate,
                "initial_filter": self.initial_filter,
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
class DenseNetBackbone(BaseModel):
    """DenseNet feature extractor. Returns ``[stem, b1..bN]`` feature maps."""

    KMODELS_CONFIG = DENSENET_CONFIG
    KMODELS_WEIGHTS = DENSENET_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        return DenseNet

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
        transfer_densenet_weights(keras_model, state_dict)

    def __init__(
        self,
        num_blocks=(6, 12, 24, 16),
        growth_rate=32,
        initial_filter=64,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="DenseNetBackbone",
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
        features = _densenet_features(
            x,
            num_blocks=num_blocks,
            growth_rate=growth_rate,
            initial_filter=initial_filter,
            channels_axis=channels_axis,
            data_format=data_format,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.num_blocks = num_blocks
        self.growth_rate = growth_rate
        self.initial_filter = initial_filter
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_blocks": self.num_blocks,
                "growth_rate": self.growth_rate,
                "initial_filter": self.initial_filter,
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
