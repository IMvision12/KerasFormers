import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import VGG_CONFIG, VGG_WEIGHTS
from .convert_vgg_torch_to_keras import transfer_vgg_weights


def vgg_block(
    inputs,
    num_filters,
    channels_axis,
    data_format,
    batch_norm=False,
):
    """Stack of Conv2D / [BN] / ReLU and MaxPool layers per the VGG recipe.

    ``num_filters`` is a list mixing ints (filter counts) and ``"M"`` markers
    for MaxPooling. Returns ``(x, features)`` where ``features`` is the list of
    tensors collected at each "M" boundary (i.e. before each MaxPool).
    """
    x = inputs
    layer_idx = 0
    features = []

    for v in num_filters:
        if v == "M":
            features.append(x)
            x = layers.MaxPooling2D(
                pool_size=2,
                strides=2,
                data_format=data_format,
                name=f"Max_Pool_{layer_idx}",
            )(x)
            layer_idx += 1
        else:
            x = layers.Conv2D(
                v,
                3,
                padding="same",
                data_format=data_format,
                name=f"conv2d_{layer_idx}",
            )(x)
            layer_idx += 1

            if batch_norm:
                x = layers.BatchNormalization(
                    axis=channels_axis,
                    momentum=0.9,
                    epsilon=1e-5,
                    name=f"batchnorm_{layer_idx}",
                )(x)
                layer_idx += 1

            x = layers.ReLU(name=f"relu_{layer_idx}")(x)
            layer_idx += 1

    return x, features


def _vgg_features(
    inputs,
    *,
    num_filters,
    batch_norm,
    data_format,
    channels_axis,
):
    """Convolutional stack + classification-head pre-logit convs.

    Returns ``[feat_at_each_M..., post_pre_logits]``.
    """
    x, features = vgg_block(
        inputs,
        num_filters,
        batch_norm=batch_norm,
        channels_axis=channels_axis,
        data_format=data_format,
    )

    x = layers.Conv2D(4096, 7, data_format=data_format, name="conv_fc1")(x)
    x = layers.ReLU(name="relu_fc1")(x)
    x = layers.Dropout(0.5, name="dropout_fc1")(x)
    x = layers.Conv2D(4096, 1, data_format=data_format, name="conv_fc2")(x)
    x = layers.ReLU(name="relu_fc2")(x)
    x = layers.Dropout(0.5, name="dropout_fc2")(x)

    features.append(x)
    return features


@keras.saving.register_keras_serializable(package="kmodels")
class VGGClassify(BaseModel):
    """VGG classifier (timm-ported).

    Reference:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (ICLR 2015)

    Construction:

    >>> VGGClassify.from_weights("vgg16_tv_in1k")
    >>> VGGClassify.from_weights("timm:timm/vgg16.tv_in1k")
    """

    KMODELS_CONFIG = VGG_CONFIG
    KMODELS_WEIGHTS = VGG_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_vgg_weights(keras_model, state_dict)

    def __init__(
        self,
        num_filters=None,
        batch_norm=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="VGGClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        if num_filters is None:
            raise ValueError("`num_filters` must be provided.")

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
        features = _vgg_features(
            x,
            num_filters=num_filters,
            batch_norm=batch_norm,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.Dropout(rate=0, name="dropout")(x)
        x = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.num_filters = num_filters
        self.batch_norm = batch_norm
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
                "num_filters": self.num_filters,
                "batch_norm": self.batch_norm,
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
class VGGModel(BaseModel):
    """VGG trunk returning the final feature map ``(B, H, W, C)``."""

    KMODELS_CONFIG = VGG_CONFIG
    KMODELS_WEIGHTS = VGG_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = VGGClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_vgg_weights(keras_model, state_dict)

    def __init__(
        self,
        num_filters=None,
        batch_norm=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="VGGModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        if num_filters is None:
            raise ValueError("`num_filters` must be provided.")

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
        features = _vgg_features(
            x,
            num_filters=num_filters,
            batch_norm=batch_norm,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features[-1], name=name, **kwargs)

        self.num_filters = num_filters
        self.batch_norm = batch_norm
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "batch_norm": self.batch_norm,
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
class VGGBackbone(BaseModel):
    """VGG feature extractor. Returns ``[stage1..stage5, post_pre_logits]`` maps."""

    KMODELS_CONFIG = VGG_CONFIG
    KMODELS_WEIGHTS = VGG_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = VGGClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_vgg_weights(keras_model, state_dict)

    def __init__(
        self,
        num_filters=None,
        batch_norm=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="VGGBackbone",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        if num_filters is None:
            raise ValueError("`num_filters` must be provided.")

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
        features = _vgg_features(
            x,
            num_filters=num_filters,
            batch_norm=batch_norm,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.num_filters = num_filters
        self.batch_norm = batch_norm
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "batch_norm": self.batch_norm,
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
