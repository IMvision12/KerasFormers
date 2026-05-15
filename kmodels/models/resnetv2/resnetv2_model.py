import keras
import numpy as np
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer, StochasticDepth
from kmodels.models.resnetv2.resnetv2_layers import StdConv2D
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import RESNETV2_MODEL_CONFIG, RESNETV2_WEIGHT_CONFIG
from .convert_resnetv2_torch_to_keras import transfer_resnetv2_weights


def make_divisible(v, divisor=8):
    """Round ``v`` to the nearest multiple of ``divisor``, never below 90% of ``v``.

    Args:
        v: Numeric value to round (typically a channel count).
        divisor: Multiple to round to. Defaults to ``8``.

    Returns:
        Int channel count that is a multiple of ``divisor`` and at least
        90% of the original ``v``.
    """
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_block(
    x,
    filters,
    kernel_size,
    data_format,
    strides=1,
    padding="same",
    use_bias=False,
    name=None,
):
    """Weight-standardized Conv2D with explicit zero-pad on strided convs.

    Args:
        x: Input Keras tensor.
        filters: Number of output filters for the convolution.
        kernel_size: Size of the convolution kernel.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        strides: Stride of the convolution.
        padding: Padding mode passed to :class:`StdConv2D` when no explicit
            zero-padding is applied.
        use_bias: Whether the convolution uses a bias term.
        name: Optional name for the convolution layer.

    Returns:
        Output tensor after weight-standardized convolution.
    """
    if strides > 1:
        pad = kernel_size // 2
        x = layers.ZeroPadding2D(padding=(pad, pad))(x)
        padding = "valid"

    x = StdConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        data_format=data_format,
        name=name,
    )(x)
    return x


def preact_bottleneck(
    x,
    filters,
    data_format,
    channels_axis,
    strides=1,
    downsample=False,
    drop_path_rate=0.0,
    block_prefix=None,
    bottleneck_ratio=0.25,
):
    """Pre-activation bottleneck used by BiT / ResNetV2.

    Args:
        x: Input Keras tensor.
        filters: Number of output filters for the block.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Int axis for the channel dimension.
        strides: Stride for the 3x3 convolution.
        downsample: Whether to project the shortcut to the new spatial /
            channel shape.
        drop_path_rate: Stochastic-depth drop probability. ``0`` disables.
        block_prefix: Name prefix for layers in the block.
        bottleneck_ratio: Reduction factor for the bottleneck channels.

    Returns:
        Output tensor of the residual block.
    """
    shortcut = x
    mid_channels = make_divisible(filters * bottleneck_ratio)

    preact = layers.GroupNormalization(
        axis=channels_axis, name=f"{block_prefix}_groupnorm_1"
    )(x)
    preact = layers.Activation("relu", name=f"{block_prefix}_relu_1")(preact)

    if downsample:
        shortcut = conv_block(
            preact,
            filters=filters,
            kernel_size=1,
            data_format=data_format,
            strides=strides,
            use_bias=False,
            name=f"{block_prefix}_downsample_conv",
        )

    x = conv_block(
        preact,
        filters=mid_channels,
        kernel_size=1,
        data_format=data_format,
        use_bias=False,
        name=f"{block_prefix}_conv_1",
    )
    x = layers.GroupNormalization(
        axis=channels_axis, name=f"{block_prefix}_groupnorm_2"
    )(x)
    x = layers.Activation("relu", name=f"{block_prefix}_relu_2")(x)

    x = conv_block(
        x,
        filters=mid_channels,
        kernel_size=3,
        data_format=data_format,
        strides=strides,
        use_bias=False,
        name=f"{block_prefix}_conv_2",
    )
    x = layers.GroupNormalization(
        axis=channels_axis, name=f"{block_prefix}_groupnorm_3"
    )(x)
    x = layers.Activation("relu", name=f"{block_prefix}_relu_3")(x)

    x = conv_block(
        x,
        filters=filters,
        kernel_size=1,
        data_format=data_format,
        use_bias=False,
        name=f"{block_prefix}_conv_3",
    )

    if drop_path_rate > 0:
        x = StochasticDepth(drop_path_rate)(x)

    x = layers.Add(name=f"{block_prefix}_add")([shortcut, x])
    return x


def resnetv2_backbone_feature(
    inputs,
    block_repeats,
    filters,
    width_factor,
    stem_width,
    drop_path_rate,
    data_format,
    channels_axis,
    return_stages=False,
):
    """ResNetV2 stem + stages, returning the final stage feature map.

    Args:
        inputs: Input image tensor.
        block_repeats: Number of pre-activation bottlenecks per stage.
        filters: Base filter count per stage.
        width_factor: Multiplier applied to channel counts (BiT widening).
        stem_width: Base width of the stem convolution before width scaling.
        drop_path_rate: Maximum stochastic-depth drop probability; linearly
            interpolated across blocks.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Int axis for the channel dimension.
        return_stages: If True, return a list of per-stage feature maps
            (one tensor per ResNetV2 stage). If False (default), return
            only the final stage map.

    Returns:
        Final stage feature tensor (pre final GroupNorm), or a list of
        per-stage feature maps when ``return_stages=True``.
    """
    x = conv_block(
        inputs,
        filters=make_divisible(stem_width * width_factor),
        kernel_size=7,
        data_format=data_format,
        strides=2,
        use_bias=False,
        name="stem_conv",
    )
    x = layers.ZeroPadding2D(data_format=data_format, padding=(1, 1))(x)
    x = layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        data_format=data_format,
        padding="valid",
        name="stem_maxpool",
    )(x)

    dpr = list(np.linspace(0.0, drop_path_rate, sum(block_repeats)))
    block_idx = 0
    stages = []
    for stage_idx, num_blocks in enumerate(block_repeats):
        nb_channels_stage = make_divisible(filters[stage_idx] * width_factor)
        for block_idx_in_stage in range(num_blocks):
            block_prefix = f"stages_{stage_idx}_blocks_{block_idx_in_stage}"
            x = preact_bottleneck(
                x,
                filters=nb_channels_stage,
                data_format=data_format,
                channels_axis=channels_axis,
                strides=2 if (stage_idx > 0 and block_idx_in_stage == 0) else 1,
                downsample=block_idx_in_stage == 0,
                drop_path_rate=dpr[block_idx],
                block_prefix=block_prefix,
            )
            block_idx += 1
        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class ResNetV2Model(BaseModel):
    """ResNetV2 trunk returning the final stage feature map ``(B, H, W, C)``.

    Output is the raw final-stage feature map (pre final GroupNorm).
    :class:`ResNetV2Classify` composes this model and applies the final
    GroupNorm + ReLU + GAP + Dense head to produce logits.

    Reference:
    - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
    - [Big Transfer (BiT)](https://arxiv.org/abs/1912.11370)
    """

    KMODELS_CONFIG = {
        variant: RESNETV2_MODEL_CONFIG[meta["model"]]
        for variant, meta in RESNETV2_WEIGHT_CONFIG.items()
    }
    KMODELS_WEIGHTS = RESNETV2_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = ResNetV2Classify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_resnetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        block_repeats=(3, 4, 6, 3),
        filters=(256, 512, 1024, 2048),
        width_factor=1,
        stem_width=64,
        drop_path_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        as_backbone=False,
        name="ResNetV2Model",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "drop_rate", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

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
        x = resnetv2_backbone_feature(
            x,
            block_repeats=block_repeats,
            filters=filters,
            width_factor=width_factor,
            stem_width=stem_width,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.block_repeats = block_repeats
        self.filters = filters
        self.width_factor = width_factor
        self.stem_width = stem_width
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "block_repeats": self.block_repeats,
                "filters": self.filters,
                "width_factor": self.width_factor,
                "stem_width": self.stem_width,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "as_backbone": self.as_backbone,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class ResNetV2Classify(BaseModel):
    """
    Instantiates a ResNetV2 / BiT classifier (timm-ported).

    Wraps a :class:`ResNetV2Model` backbone and applies the final
    GroupNorm + ReLU + GlobalAveragePooling + optional Dropout + Dense
    head to produce class logits.

    Reference:
    - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
    - [Big Transfer (BiT)](https://arxiv.org/abs/1912.11370)

    Construction:

    >>> ResNetV2Classify.from_weights("resnetv2_50x1_bit_goog_in21k_ft_in1k")
    >>> ResNetV2Classify.from_weights("timm:timm/resnetv2_50x1_bit.goog_in21k_ft_in1k")
    """

    KMODELS_CONFIG = {
        variant: RESNETV2_MODEL_CONFIG[meta["model"]]
        for variant, meta in RESNETV2_WEIGHT_CONFIG.items()
    }
    KMODELS_WEIGHTS = RESNETV2_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_resnetv2_weights(keras_model, state_dict)

    def __init__(
        self,
        block_repeats=(3, 4, 6, 3),
        filters=(256, 512, 1024, 2048),
        width_factor=1,
        stem_width=64,
        drop_rate=0.0,
        drop_path_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ResNetV2Classify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        backbone = ResNetV2Model(
            block_repeats=block_repeats,
            filters=filters,
            width_factor=width_factor,
            stem_width=stem_width,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            input_shape=input_shape,
            name=f"{name}_backbone",
        )

        x = layers.GroupNormalization(axis=channels_axis, name="groupnorm")(
            backbone.output
        )
        x = layers.Activation("relu", name="relu")(x)
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(x)
        if drop_rate > 0:
            x = layers.Dropout(drop_rate, name="dropout")(x)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.block_repeats = block_repeats
        self.filters = filters
        self.width_factor = width_factor
        self.stem_width = stem_width
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
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
                "block_repeats": self.block_repeats,
                "filters": self.filters,
                "width_factor": self.width_factor,
                "stem_width": self.stem_width,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
