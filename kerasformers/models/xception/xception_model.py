import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kerasformers.base import BaseModel
from kerasformers.layers import ImageNormalizationLayer
from kerasformers.weight_utils import copy_weights_by_path_suffix

from .config import XCEPTION_MODEL_CONFIG, XCEPTION_WEIGHT_CONFIG
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


@keras.saving.register_keras_serializable(package="kerasformers")
class XceptionModel(BaseModel):
    """Instantiates the Xception backbone.

    Xception ("extreme Inception") replaces standard convolutions with
    depthwise-separable convolutions throughout an entry flow, a
    middle flow of 8 repeated 728-channel blocks, and an exit flow, with
    residual skip connections around each block. Output is the last
    layer output before the classifier head: the final 2048-channel
    feature map ``(B, H, W, C)`` from the exit flow.
    :class:`XceptionClassify` composes this model and attaches a
    GlobalAveragePooling2D + Dense head to produce logits.

    Note: This is the *original* Keras Xception (Chollet 2017),
    warm-started from ``keras.applications.Xception``. timm's
    xception41/65/71 families use a different *Aligned Xception*
    backbone that is not implemented in this module.

    References:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps ``[entry, middle, exit]`` (one per
            flow). Defaults to `False`.
        image_size: Integer, square input resolution used to validate the
            input shape. Defaults to `299`.
        include_normalization: Boolean, whether to prepend an
            :class:`~kerasformers.layers.ImageNormalizationLayer` at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'`, `'inception'` (default),
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_shape: Optional tuple specifying the shape of the input
            data. If `None`, derived from ``image_size`` and the active
            Keras data format. Defaults to `None`.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        name: String, the name of the model.
            Defaults to `"XceptionModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = {
        variant: XCEPTION_MODEL_CONFIG[meta["model"]]
        for variant, meta in XCEPTION_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = XCEPTION_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = XceptionClassify.from_weights(variant, skip_mismatch=skip_mismatch)
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


@keras.saving.register_keras_serializable(package="kerasformers")
class XceptionClassify(BaseModel):
    """Instantiates the Xception classifier.

    This classifier wraps an :class:`XceptionModel` backbone and
    attaches a GlobalAveragePooling2D + Dense head to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`XceptionModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    Note: This is the *original* Keras Xception (Chollet 2017),
    warm-started from ``keras.applications.Xception``. timm's
    xception41/65/71 families use a different *Aligned Xception*
    backbone that is not implemented in this module.

    References:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

    Args:
        image_size: Integer, square input resolution used to validate the
            input shape. Defaults to `299`.
        include_normalization: Boolean, whether to prepend an
            :class:`~kerasformers.layers.ImageNormalizationLayer` at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'`, `'inception'` (default),
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_shape: Optional tuple specifying the shape of the input
            data. If `None`, derived from ``image_size`` and the active
            Keras data format. Defaults to `None`.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        num_classes: Integer, the number of output classes for
            classification. Defaults to `1000`.
        classifier_activation: String or callable, activation function
            for the final Dense layer. Use `"linear"` to return raw
            logits or `"softmax"` to return class probabilities.
            Defaults to `"linear"`.
        name: String, the name of the model. The internal backbone is
            named `f"{name}_backbone"`. Defaults to `"XceptionClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = {
        variant: XCEPTION_MODEL_CONFIG[meta["model"]]
        for variant, meta in XCEPTION_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = XCEPTION_WEIGHT_CONFIG
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
