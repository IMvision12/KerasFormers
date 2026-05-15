from typing import Optional

import keras
from keras import layers

from kmodels.models.resnet.resnet_model import (
    ResNetClassify,
    ResNetModel,
    conv_block,
    squeeze_excitation_block,
)
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import RESNEXT_MODEL_CONFIG, RESNEXT_WEIGHT_CONFIG


def resnext_block(
    x: layers.Layer,
    filters: int,
    channels_axis,
    data_format,
    strides: int = 1,
    groups: int = 32,
    width_factor: int = 2,
    downsample: bool = False,
    senet: bool = False,
    block_name: Optional[str] = None,
) -> layers.Layer:
    """ResNeXt block with group convolutions.

    Args:
        x: Input Keras layer.
        filters: Number of filters for the block.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        strides: Stride for the main convolution layer.
        groups: Number of groups for grouped convolution.
        width_factor: Factor to determine width for grouped convolution.
        downsample: Whether to downsample the input.
        senet: Whether to apply SE block.
        block_name: Optional name for layers in the block.

    Returns:
        Output tensor for the block.
    """
    residual = x
    expansion = 4
    width = filters * width_factor

    x = conv_block(
        x,
        width,
        kernel_size=1,
        strides=1,
        name=f"{block_name}_conv1",
        bn_name=f"{block_name}_batchnorm1",
        channels_axis=channels_axis,
        data_format=data_format,
    )
    group_width = width // groups
    x = conv_block(
        x,
        width,
        kernel_size=3,
        strides=strides,
        groups=groups,
        group_width=group_width,
        name=f"{block_name}_conv2",
        bn_name=f"{block_name}_batchnorm2",
        channels_axis=channels_axis,
        data_format=data_format,
    )
    x = conv_block(
        x,
        filters * expansion,
        kernel_size=1,
        use_relu=False,
        name=f"{block_name}_conv3",
        bn_name=f"{block_name}_batchnorm3",
        channels_axis=channels_axis,
        data_format=data_format,
    )

    if senet:
        x = squeeze_excitation_block(
            x, data_format=data_format, name=f"{block_name}_se"
        )

    if (
        downsample
        or strides != 1
        or x.shape[channels_axis] != residual.shape[channels_axis]
    ):
        residual = conv_block(
            residual,
            filters * expansion,
            kernel_size=1,
            strides=strides,
            use_relu=False,
            name=f"{block_name}_downsample_conv",
            bn_name=f"{block_name}_downsample_batchnorm",
            channels_axis=channels_axis,
            data_format=data_format,
        )

    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    return x


@keras.saving.register_keras_serializable(package="kmodels")
class ResNeXtModel(ResNetModel):
    """ResNeXt trunk returning the final stage feature map ``(B, H, W, C)``."""

    KMODELS_CONFIG = RESNEXT_MODEL_CONFIG
    KMODELS_WEIGHTS = RESNEXT_WEIGHT_CONFIG

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = ResNeXtClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    def __init__(
        self,
        block_fn=resnext_block,
        block_repeats=[3, 4, 6, 3],
        filters=[64, 128, 256, 512],
        groups=32,
        width_factor=2,
        as_backbone=False,
        name="ResNeXtModel",
        **kwargs,
    ):
        super().__init__(
            block_fn=block_fn,
            block_repeats=block_repeats,
            filters=filters,
            groups=groups,
            width_factor=width_factor,
            as_backbone=as_backbone,
            name=name,
            **kwargs,
        )


@keras.saving.register_keras_serializable(package="kmodels")
class ResNeXtClassify(ResNetClassify):
    """ResNeXt (grouped-convolution ResNet) classifier.

    Same skeleton as :class:`ResNetClassify` but composes a
    :class:`ResNeXtModel` backbone with :func:`resnext_block` and the
    cardinality knobs (``groups`` / ``width_factor``). Variant ids and
    release weights live in :data:`RESNEXT_MODEL_CONFIG` / :data:`RESNEXT_WEIGHT_CONFIG`.

    >>> ResNeXtClassify.from_weights("resnext50_32x4d_a1_in1k")
    >>> ResNeXtClassify.from_weights("timm:timm/resnext50_32x4d.a1_in1k")
    """

    KMODELS_CONFIG = RESNEXT_MODEL_CONFIG
    KMODELS_WEIGHTS = RESNEXT_WEIGHT_CONFIG

    def __init__(
        self,
        block_fn=resnext_block,
        block_repeats=[3, 4, 6, 3],
        filters=[64, 128, 256, 512],
        groups=32,
        senet=False,
        width_factor=2,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ResNeXtClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = ResNeXtModel(
            block_fn=block_fn,
            block_repeats=block_repeats,
            filters=filters,
            groups=groups,
            senet=senet,
            width_factor=width_factor,
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
            kernel_initializer="zeros",
            name="predictions",
        )(x)

        super(ResNetClassify, self).__init__(
            inputs=backbone.input, outputs=out, name=name, **kwargs
        )

        self.block_fn = block_fn
        self.block_repeats = block_repeats
        self.filters = filters
        self.groups = groups
        self.senet = senet
        self.width_factor = width_factor
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
