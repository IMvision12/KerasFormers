import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import XCEPTION_CONFIG, XCEPTION_WEIGHTS
from .convert_xception_org_keras_to_keras import transfer_xception_weights


def conv_block(
    x,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="same",
    separable=False,
    use_activation=True,
    use_preactivation=False,
    use_bias=False,
):
    """Standard or separable Conv -> BatchNorm with optional pre / post ReLU."""
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    x = layers.Activation("relu")(x) if use_preactivation else x

    conv_layer = layers.SeparableConv2D if separable else layers.Conv2D
    x = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        data_format=data_format,
    )(x)

    x = layers.BatchNormalization(axis=channels_axis)(x)
    x = layers.Activation("relu")(x) if use_activation else x
    return x


def entry_flow(x):
    """Entry flow (blocks 1-4)."""
    x = conv_block(x, 32, (3, 3), strides=(2, 2), padding="valid")
    x = conv_block(x, 64, (3, 3), padding="valid")

    residual = conv_block(x, 128, (1, 1), strides=(2, 2), use_activation=False)

    x = conv_block(x, 128, (3, 3), separable=True)
    x = conv_block(x, 128, (3, 3), separable=True, use_activation=False)
    x = layers.MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        data_format=keras.config.image_data_format(),
        padding="same",
    )(x)
    x = layers.add([x, residual])

    residual = conv_block(
        x, 256, (1, 1), strides=(2, 2), use_bias=False, use_activation=False
    )
    x = conv_block(x, 256, (3, 3), use_preactivation=True, separable=True)
    x = conv_block(x, 256, (3, 3), use_activation=False, separable=True)
    x = layers.MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        data_format=keras.config.image_data_format(),
        padding="same",
    )(x)
    x = layers.add([x, residual])

    residual = conv_block(x, 728, (1, 1), strides=(2, 2), use_activation=False)
    x = conv_block(x, 728, (3, 3), separable=True, use_preactivation=True)
    x = conv_block(x, 728, (3, 3), separable=True, use_activation=False)
    x = layers.MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        data_format=keras.config.image_data_format(),
        padding="same",
    )(x)
    x = layers.add([x, residual])

    return x


def middle_flow(x):
    """Middle flow (8 repeated 728-channel blocks)."""
    for i in range(8):
        residual = x
        x = conv_block(x, 728, (3, 3), separable=True, use_preactivation=True)
        x = conv_block(x, 728, (3, 3), separable=True)
        x = conv_block(x, 728, (3, 3), separable=True, use_activation=False)
        x = layers.add([x, residual])
    return x


def exit_flow(x):
    """Exit flow (blocks 13-14)."""
    residual = conv_block(x, 1024, (1, 1), strides=(2, 2), use_activation=False)

    x = conv_block(x, 728, (3, 3), separable=True, use_preactivation=True)
    x = conv_block(x, 1024, (3, 3), separable=True, use_activation=False)
    x = layers.MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        data_format=keras.config.image_data_format(),
        padding="same",
    )(x)
    x = layers.add([x, residual])

    x = conv_block(x, 1536, (3, 3), separable=True)
    x = conv_block(x, 2048, (3, 3), separable=True)

    return x


def _xception_features(inputs):
    """Xception entry / middle / exit flows, returns ``[entry, middle, exit]``."""
    features = []
    x = entry_flow(inputs)
    features.append(x)
    x = middle_flow(x)
    features.append(x)
    x = exit_flow(x)
    features.append(x)
    return features


@keras.saving.register_keras_serializable(package="kmodels")
class Xception(BaseModel):
    """Original-Keras Xception classifier.

    Reference:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) (CVPR 2017)

    Note: This is the *original* Keras Xception (Chollet 2017), warm-started
    from ``keras.applications.Xception``. timm's xception41/65/71 families
    use a different *Aligned Xception* backbone that is not implemented
    in this module.

    Construction:

    >>> Xception.from_weights("xception_in1k")
    """

    KMODELS_CONFIG = XCEPTION_CONFIG
    KMODELS_WEIGHTS = XCEPTION_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_xception_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="Xception",
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
        features = _xception_features(x)
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
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
class XceptionBackbone(BaseModel):
    """Xception feature extractor. Returns ``[entry, middle, exit]`` (3 maps)."""

    KMODELS_CONFIG = XCEPTION_CONFIG
    KMODELS_WEIGHTS = XCEPTION_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        return Xception

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
        transfer_xception_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        name="XceptionBackbone",
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
        features = _xception_features(x)

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
