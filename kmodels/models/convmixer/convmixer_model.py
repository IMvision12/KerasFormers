import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import CONVMIXER_MODEL_CONFIG, CONVMIXER_WEIGHT_CONFIG
from .convert_convmixer_torch_to_keras import transfer_convmixer_weights


def convmixer_block(
    x, filters, kernel_size, activation, channels_axis, data_format, name
):
    """A building block for the ConvMixer architecture.

    Args:
        x: input tensor.
        filters: int, the number of output filters for the convolution layers.
        kernel_size: int, the size of the convolution kernel.
        activation: string, name of the activation function to be applied within
            the Conv2D layers (e.g., 'gelu', 'relu').
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        name: string, block name.

    Returns:
        Output tensor for the block.
    """
    inputs = x
    x = layers.DepthwiseConv2D(
        kernel_size,
        1,
        padding="same",
        use_bias=True,
        activation=activation,
        data_format=data_format,
        name=f"{name}_depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_1"
    )(x)

    x = layers.Add(name=f"{name}_add")([inputs, x])

    x = layers.Conv2D(
        filters,
        1,
        1,
        activation=activation,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv2d",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_batchnorm_2"
    )(x)

    return x


def convmixer_backbone_feature(
    inputs,
    *,
    dim,
    depth,
    kernel_size,
    patch_size,
    activation,
    data_format,
    channels_axis,
    return_stages=False,
):
    """Stem + N ConvMixer blocks, returning the final feature map.

    Args:
        inputs: Input image tensor (post-normalization).
        dim: Channel dimension carried throughout the model.
        depth: Total number of ConvMixer blocks stacked after the stem.
        kernel_size: Depthwise convolution kernel size inside each block.
        patch_size: Stride/kernel of the patch-embedding stem convolution.
        activation: Activation name applied inside conv layers (e.g. ``"gelu"``).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Axis index of the channels dimension.
        return_stages: If True, return a singleton list ``[final]`` (ConvMixer
            has no natural multi-stage hierarchy — all blocks share the same
            spatial resolution and channel count). If False (default), return
            the final feature map directly.

    Returns:
        Final feature map ``(B, H, W, C)``, or a singleton list
        ``[final]`` when ``return_stages=True``.
    """
    x = layers.Conv2D(
        dim,
        kernel_size=patch_size,
        strides=patch_size,
        use_bias=True,
        activation=activation,
        data_format=data_format,
        name="stem_conv2d",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_batchnorm"
    )(x)

    for i in range(depth):
        x = convmixer_block(
            x,
            dim,
            kernel_size,
            activation,
            channels_axis,
            data_format,
            f"mixer_block_{i}",
        )

    if return_stages:
        return [x]
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class ConvMixerModel(BaseModel):
    """ConvMixer backbone — the main feature extractor.

    Returns the final feature map ``(B, H, W, C)``. This is the last layer
    output before the classifier head. :class:`ConvMixerClassify` composes
    this model and attaches GlobalAveragePooling + Dense to produce
    class logits.

    Reference:
    - [Patches Are All You Need?](https://arxiv.org/abs/2201.09792) (OpenReview 2022)

    Construction:

    >>> ConvMixerModel.from_weights("convmixer_768_32_in1k")
    >>> ConvMixerModel.from_weights("timm:timm/convmixer_768_32.in1k")
    """

    BASE_MODEL_CONFIG = {
        variant: CONVMIXER_MODEL_CONFIG[meta["model"]]
        for variant, meta in CONVMIXER_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = CONVMIXER_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = ConvMixerClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_convmixer_weights(keras_model, state_dict)

    def __init__(
        self,
        dim=768,
        depth=32,
        kernel_size=7,
        patch_size=7,
        activation="gelu",
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        as_backbone=False,
        name="ConvMixerModel",
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
        x = convmixer_backbone_feature(
            x,
            dim=dim,
            depth=depth,
            kernel_size=kernel_size,
            patch_size=patch_size,
            activation=activation,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "depth": self.depth,
                "patch_size": self.patch_size,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
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
class ConvMixerClassify(BaseModel):
    """ConvMixer image classifier — :class:`ConvMixerModel` + GAP + Dense head.

    Wraps a :class:`ConvMixerModel` backbone and attaches GlobalAveragePooling
    and a single Dense layer on the final feature map to produce class logits.

    Reference:
    - [Patches Are All You Need?](https://arxiv.org/abs/2201.09792) (OpenReview 2022)

    Construction:

    >>> ConvMixerClassify.from_weights("convmixer_768_32_in1k")
    >>> ConvMixerClassify.from_weights("timm:timm/convmixer_768_32.in1k")
    """

    BASE_MODEL_CONFIG = {
        variant: CONVMIXER_MODEL_CONFIG[meta["model"]]
        for variant, meta in CONVMIXER_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = CONVMIXER_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_convmixer_weights(keras_model, state_dict)

    def __init__(
        self,
        dim=768,
        depth=32,
        kernel_size=7,
        patch_size=7,
        activation="gelu",
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ConvMixerClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = ConvMixerModel(
            dim=dim,
            depth=depth,
            kernel_size=kernel_size,
            patch_size=patch_size,
            activation=activation,
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
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.activation = activation
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
                "dim": self.dim,
                "depth": self.depth,
                "patch_size": self.patch_size,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
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
