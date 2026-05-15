import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import DENSENET_MODEL_CONFIG, DENSENET_WEIGHT_CONFIG
from .convert_densenet_torch_to_keras import transfer_densenet_weights


def conv_block(
    x,
    growth_rate,
    expansion_ratio,
    channels_axis,
    data_format,
    name,
):
    """Single conv layer inside a DenseNet dense block.

    Args:
        x: Input feature tensor.
        growth_rate: Output channel count of the 3x3 conv.
        expansion_ratio: Multiplier on ``growth_rate`` for the 1x1 bottleneck.
        channels_axis: Axis index of the channels dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        name: Name prefix for sub-layers.

    Returns:
        Tensor with ``growth_rate`` channels to be concatenated with prior layers.
    """
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
    """Dense block: stack of conv blocks with channel-wise concatenation.

    Args:
        x: Input feature tensor entering the block.
        num_layers: Number of internal conv blocks (a.k.a. dense layers).
        growth_rate: Per-layer channel growth.
        channels_axis: Axis index of the channels dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        name: Name prefix for sub-layers.

    Returns:
        Concatenated feature tensor whose channel count is
        ``x.channels + num_layers * growth_rate``.
    """
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
    """Reduce channels by ``reduction`` and 2x downsample spatially.

    Args:
        x: Input feature tensor from the preceding dense block.
        reduction: Channel reduction factor (e.g. ``0.5`` halves the channels).
        channels_axis: Axis index of the channels dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        name: Name prefix for sub-layers.

    Returns:
        Down-projected tensor with halved spatial resolution.
    """
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


def densenet_backbone_feature(
    inputs,
    *,
    num_blocks,
    growth_rate,
    initial_filter,
    channels_axis,
    data_format,
    return_stages=False,
):
    """Stem + N dense blocks + final BN/ReLU, returning the final feature map.

    Args:
        inputs: Input image tensor (post-normalization).
        num_blocks: Tuple of layer counts per dense block (e.g. ``(6, 12, 24, 16)``).
        growth_rate: Per-layer channel growth inside each dense block.
        initial_filter: Channel count for the 7x7 stem convolution.
        channels_axis: Axis index of the channels dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_stages: If True, return a list of per-stage feature maps (one
            per dense block; the final stage is the post BN+ReLU output). If
            False (default), return only the final feature map.

    Returns:
        Final feature map ``(B, H, W, C)`` with BN+ReLU applied, or a list of
        per-stage feature maps when ``return_stages=True``.
    """
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

    stages = []
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
            stages.append(x)

    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name="final_batchnorm",
    )(x)
    x = layers.ReLU(name="final_relu")(x)
    stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class DenseNetModel(BaseModel):
    """DenseNet backbone — the main feature extractor.

    Returns the final feature map ``(B, H, W, C)`` (post BN+ReLU). This is
    the last layer output before the classifier head. :class:`DenseNetClassify`
    composes this model and attaches GlobalAveragePooling + Dense to produce
    class logits.

    Reference:
    - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

    Construction:

    >>> DenseNetModel.from_weights("densenet121_tv_in1k")
    >>> DenseNetModel.from_weights("timm:timm/densenet121.tv_in1k")
    """

    KMODELS_CONFIG = {
        variant: DENSENET_MODEL_CONFIG[meta["model"]]
        for variant, meta in DENSENET_WEIGHT_CONFIG.items()
    }
    KMODELS_WEIGHTS = DENSENET_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = DenseNetClassify.from_weights(variant)
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
        as_backbone=False,
        name="DenseNetModel",
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
        x = densenet_backbone_feature(
            x,
            num_blocks=num_blocks,
            growth_rate=growth_rate,
            initial_filter=initial_filter,
            channels_axis=channels_axis,
            data_format=data_format,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.num_blocks = num_blocks
        self.growth_rate = growth_rate
        self.initial_filter = initial_filter
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

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
                "as_backbone": self.as_backbone,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class DenseNetClassify(BaseModel):
    """DenseNet image classifier — :class:`DenseNetModel` + GAP + Dense head.

    Wraps a :class:`DenseNetModel` backbone and attaches GlobalAveragePooling
    and a single Dense layer on the final feature map to produce class logits.

    Reference:
    - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

    Construction:

    >>> DenseNetClassify.from_weights("densenet121_tv_in1k")
    >>> DenseNetClassify.from_weights("timm:timm/densenet121.tv_in1k")
    """

    KMODELS_CONFIG = {
        variant: DENSENET_MODEL_CONFIG[meta["model"]]
        for variant, meta in DENSENET_WEIGHT_CONFIG.items()
    }
    KMODELS_WEIGHTS = DENSENET_WEIGHT_CONFIG
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
        name="DenseNetClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = DenseNetModel(
            num_blocks=num_blocks,
            growth_rate=growth_rate,
            initial_filter=initial_filter,
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
