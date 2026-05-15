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
    """Standard or separable Conv -> BatchNorm with optional pre / post ReLU.

    Args:
        x: Input feature tensor.
        filters: Output channel count of the convolution.
        kernel_size: Convolution kernel size (int or tuple).
        strides: Convolution strides.
        padding: ``"same"`` or ``"valid"`` padding mode.
        separable: If True, use ``SeparableConv2D`` instead of ``Conv2D``.
        use_activation: If True, apply ReLU after BatchNorm (post-activation).
        use_preactivation: If True, apply ReLU before the convolution.
        use_bias: Whether the convolution uses a bias term.

    Returns:
        Tensor produced by the configured Conv + BN (+ activation) chain.
    """
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
    """Entry flow (blocks 1-4).

    Args:
        x: Input image tensor (post-normalization).

    Returns:
        Feature tensor at 728 channels and 1/16 input spatial resolution.
    """
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
    """Middle flow (8 repeated 728-channel blocks).

    Args:
        x: Feature tensor from the entry flow (728 channels).

    Returns:
        Tensor with the same shape as ``x`` after 8 residual separable blocks.
    """
    for i in range(8):
        residual = x
        x = conv_block(x, 728, (3, 3), separable=True, use_preactivation=True)
        x = conv_block(x, 728, (3, 3), separable=True)
        x = conv_block(x, 728, (3, 3), separable=True, use_activation=False)
        x = layers.add([x, residual])
    return x


def exit_flow(x):
    """Exit flow (blocks 13-14).

    Args:
        x: Feature tensor from the middle flow (728 channels).

    Returns:
        Final feature tensor at 2048 channels and 1/32 input spatial resolution.
    """
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


def xception_backbone_feature(inputs, *, return_stages=False):
    """Xception entry / middle / exit flows, returns the final feature map.

    Args:
        inputs: Input image tensor (post-normalization).
        return_stages: If True, return a list of per-stage feature maps
            ``[entry, middle, exit]`` (3 stages, one per flow). If False
            (default), return only the final exit-flow feature map.

    Returns:
        Final feature map ``(B, H, W, C)`` from the exit flow, or a list of
        ``[entry, middle, exit]`` feature maps when ``return_stages=True``.
    """
    entry = entry_flow(inputs)
    middle = middle_flow(entry)
    exit_ = exit_flow(middle)
    if return_stages:
        return [entry, middle, exit_]
    return exit_


@keras.saving.register_keras_serializable(package="kmodels")
class XceptionModel(BaseModel):
    """Xception backbone — the main feature extractor.

    Returns the final feature map ``(B, H, W, C)`` from the exit flow. This
    is the last layer output before the classifier head.
    :class:`XceptionClassify` composes this model and attaches
    GlobalAveragePooling + Dense to produce class logits.

    Reference:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) (CVPR 2017)

    Note: This is the *original* Keras Xception (Chollet 2017), warm-started
    from ``keras.applications.Xception``. timm's xception41/65/71 families
    use a different *Aligned Xception* backbone that is not implemented
    in this module.

    Construction:

    >>> XceptionModel.from_weights("xception_in1k")
    """

    KMODELS_CONFIG = XCEPTION_CONFIG
    KMODELS_WEIGHTS = XCEPTION_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = XceptionClassify.from_weights(variant)
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
        as_backbone=False,
        name="XceptionModel",
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
        x = xception_backbone_feature(x, return_stages=as_backbone)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
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
class XceptionClassify(BaseModel):
    """Xception image classifier — :class:`XceptionModel` + GAP + Dense head.

    Wraps an :class:`XceptionModel` backbone and attaches
    GlobalAveragePooling and a single Dense layer on the final feature map
    to produce class logits.

    Reference:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) (CVPR 2017)

    Note: This is the *original* Keras Xception (Chollet 2017), warm-started
    from ``keras.applications.Xception``. timm's xception41/65/71 families
    use a different *Aligned Xception* backbone that is not implemented
    in this module.

    Construction:

    >>> XceptionClassify.from_weights("xception_in1k")
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
        name="XceptionClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = XceptionModel(
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
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

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
