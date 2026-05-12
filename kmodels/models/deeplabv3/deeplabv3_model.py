import keras
from keras import layers, ops, utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer

from .config import DEEPLABV3_CONFIG, DEEPLABV3_WEIGHTS


def deeplabv3_dilated_resnet_backbone(
    inputs,
    backbone_variant,
    include_normalization=False,
    normalization_mode="imagenet",
):
    """Build a dilated ResNet backbone for DeepLabV3.

    Constructs a ResNet-50 or ResNet-101 backbone with dilated
    (atrous) convolutions in the last two stages, matching the
    torchvision DeepLabV3 backbone configuration
    (``output_stride=8``).

    Args:
        inputs: Input Keras tensor.
        backbone_variant: ``"ResNet50"`` or ``"ResNet101"``.
        include_normalization: Whether to prepend
            :class:`ImageNormalizationLayer`.
        normalization_mode: Normalization preset.

    Returns:
        C5 feature tensor at 1/8 of the input spatial resolution.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    block_repeats = {
        "ResNet50": [3, 4, 6, 3],
        "ResNet101": [3, 4, 23, 3],
    }[backbone_variant]

    x = (
        ImageNormalizationLayer(mode=normalization_mode)(inputs)
        if include_normalization
        else inputs
    )

    x = layers.ZeroPadding2D(padding=3, data_format=data_format)(x)
    x = layers.Conv2D(
        64,
        7,
        strides=2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="backbone_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name="backbone_bn1"
    )(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
    x = layers.MaxPooling2D(
        pool_size=3, strides=2, padding="valid", data_format=data_format
    )(x)

    filters_list = [64, 128, 256, 512]
    dilate_stages = [False, False, True, True]
    current_dilation = 1

    for stage_idx, num_blocks in enumerate(block_repeats):
        filters = filters_list[stage_idx]
        original_stride = 2 if stage_idx > 0 else 1

        if dilate_stages[stage_idx] and stage_idx > 0:
            current_dilation *= original_stride
            stage_stride = 1
        else:
            stage_stride = original_stride

        previous_dilation = current_dilation // (
            original_stride if dilate_stages[stage_idx] and stage_idx > 0 else 1
        )

        for block_idx in range(num_blocks):
            prefix = f"backbone_layer{stage_idx + 1}_{block_idx}"

            if block_idx == 0:
                block_stride = stage_stride
                block_dilation = previous_dilation
            else:
                block_stride = 1
                block_dilation = current_dilation

            residual = x

            x = layers.Conv2D(
                filters,
                1,
                strides=1,
                padding="valid",
                use_bias=False,
                data_format=data_format,
                name=f"{prefix}_conv1",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{prefix}_bn1"
            )(x)
            x = layers.ReLU()(x)

            if block_stride > 1:
                pad_size = block_dilation
                x = layers.ZeroPadding2D(padding=pad_size, data_format=data_format)(x)
                x = layers.Conv2D(
                    filters,
                    3,
                    strides=block_stride,
                    padding="valid",
                    dilation_rate=block_dilation,
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_conv2",
                )(x)
            else:
                if block_dilation > 1:
                    pad_size = block_dilation
                    x = layers.ZeroPadding2D(padding=pad_size, data_format=data_format)(
                        x
                    )
                    x = layers.Conv2D(
                        filters,
                        3,
                        strides=1,
                        padding="valid",
                        dilation_rate=block_dilation,
                        use_bias=False,
                        data_format=data_format,
                        name=f"{prefix}_conv2",
                    )(x)
                else:
                    x = layers.Conv2D(
                        filters,
                        3,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        data_format=data_format,
                        name=f"{prefix}_conv2",
                    )(x)

            x = layers.BatchNormalization(
                axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{prefix}_bn2"
            )(x)
            x = layers.ReLU()(x)

            x = layers.Conv2D(
                filters * 4,
                1,
                strides=1,
                padding="valid",
                use_bias=False,
                data_format=data_format,
                name=f"{prefix}_conv3",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{prefix}_bn3"
            )(x)

            in_channels = residual.shape[channels_axis]
            out_channels = filters * 4
            if block_stride != 1 or in_channels != out_channels:
                if block_stride > 1:
                    residual = layers.ZeroPadding2D(padding=0, data_format=data_format)(
                        residual
                    )
                residual = layers.Conv2D(
                    out_channels,
                    1,
                    strides=block_stride,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_downsample_conv",
                )(residual)
                residual = layers.BatchNormalization(
                    axis=channels_axis,
                    epsilon=1e-5,
                    momentum=0.1,
                    name=f"{prefix}_downsample_bn",
                )(residual)

            x = layers.Add()([x, residual])
            x = layers.ReLU()(x)

    return x


def deeplabv3_aspp(x, name="classifier_0"):
    """Atrous Spatial Pyramid Pooling module (rates 12, 24, 36)."""
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    branches = []

    b0 = layers.Conv2D(
        256, 1, use_bias=False, data_format=data_format, name=f"{name}_convs_0_0"
    )(x)
    b0 = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{name}_convs_0_1"
    )(b0)
    b0 = layers.ReLU()(b0)
    branches.append(b0)

    for i, rate in enumerate([12, 24, 36], start=1):
        b = layers.Conv2D(
            256,
            3,
            padding="same",
            dilation_rate=rate,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_convs_{i}_0",
        )(x)
        b = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.1,
            name=f"{name}_convs_{i}_1",
        )(b)
        b = layers.ReLU()(b)
        branches.append(b)

    input_shape = ops.shape(x)
    if data_format == "channels_last":
        target_h, target_w = input_shape[1], input_shape[2]
    else:
        target_h, target_w = input_shape[2], input_shape[3]

    b4 = layers.GlobalAveragePooling2D(
        data_format=data_format, keepdims=True, name=f"{name}_convs_4_0"
    )(x)
    b4 = layers.Conv2D(
        256, 1, use_bias=False, data_format=data_format, name=f"{name}_convs_4_1"
    )(b4)
    b4 = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{name}_convs_4_2"
    )(b4)
    b4 = layers.ReLU()(b4)
    b4 = layers.Resizing(
        height=target_h,
        width=target_w,
        interpolation="bilinear",
        data_format=data_format,
        name=f"{name}_convs_4_upsample",
    )(b4)
    branches.append(b4)

    x = layers.Concatenate(axis=channels_axis, name=f"{name}_concat")(branches)

    x = layers.Conv2D(
        256, 1, use_bias=False, data_format=data_format, name=f"{name}_project_0"
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{name}_project_1"
    )(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    return x


def deeplabv3_classifier_head(x, num_classes, name="classifier"):
    """DeepLabV3 classifier head: 3x3 conv + BN + ReLU + 1x1 conv."""
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    x = layers.Conv2D(
        256,
        3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{name}_2"
    )(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_classes, 1, data_format=data_format, name=f"{name}_4")(x)
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class DeepLabV3Model(BaseModel):
    """DeepLabV3 dilated ResNet backbone (no segmentation head).

    Builds the dilated ResNet-50 or ResNet-101 backbone used by
    DeepLabV3 with ``output_stride=8`` and exposes the final
    2048-channel feature map. Pair with :class:`DeepLabV3Segment`
    to get the full segmentation outputs.

    Reference:
        - `Rethinking Atrous Convolution for Semantic Image
          Segmentation <https://arxiv.org/abs/1706.05587>`_

    Args:
        backbone_variant: ``"ResNet50"`` or ``"ResNet101"``.
        include_normalization: Whether to add an
            :class:`ImageNormalizationLayer` at the input.
        normalization_mode: Normalization preset (e.g. ``"imagenet"``).
        input_shape: Image input shape excluding batch dim. Defaults
            to ``(520, 520, 3)``.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    KMODELS_CONFIG = DEEPLABV3_CONFIG
    KMODELS_WEIGHTS = None
    HF_MODEL_TYPE = None

    def __init__(
        self,
        backbone_variant="ResNet50",
        include_normalization=False,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="DeepLabV3Model",
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (520, 520, 3)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        backbone_features = deeplabv3_dilated_resnet_backbone(
            img_input,
            backbone_variant=backbone_variant,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
        )

        super().__init__(
            inputs=img_input, outputs=backbone_features, name=name, **kwargs
        )

        self.backbone_variant = backbone_variant
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_variant": self.backbone_variant,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self._input_shape_val,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class DeepLabV3Segment(BaseModel):
    """DeepLabV3 full semantic segmentation model (backbone + ASPP + head).

    Composes :class:`DeepLabV3Model`, adds the ASPP module, the
    classifier head, and a bilinear upsample back to the input
    resolution. Output shape is ``(B, H, W, num_classes)`` in
    ``channels_last``.

    Reference:
        - `Rethinking Atrous Convolution for Semantic Image
          Segmentation <https://arxiv.org/abs/1706.05587>`_

    Args:
        backbone_variant: ``"ResNet50"`` or ``"ResNet101"``.
        num_classes: Number of segmentation classes.
        include_normalization: Whether to add an
            :class:`ImageNormalizationLayer` at the input.
        normalization_mode: Normalization preset (e.g. ``"imagenet"``).
        input_shape: Image input shape excluding batch dim.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    KMODELS_CONFIG = DEEPLABV3_CONFIG
    KMODELS_WEIGHTS = DEEPLABV3_WEIGHTS
    HF_MODEL_TYPE = None

    def __init__(
        self,
        backbone_variant="ResNet50",
        num_classes=21,
        include_normalization=False,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="DeepLabV3Segment",
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (520, 520, 3)

        base = DeepLabV3Model(
            backbone_variant=backbone_variant,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_model",
        )

        x = deeplabv3_aspp(base.output, name="classifier_0")
        x = deeplabv3_classifier_head(x, num_classes, name="classifier")

        x = layers.Resizing(
            height=input_shape[0],
            width=input_shape[1],
            interpolation="bilinear",
            name="final_upsampling",
        )(x)

        super().__init__(inputs=base.input, outputs=x, name=name, **kwargs)

        self.backbone_variant = backbone_variant
        self.num_classes = num_classes
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_variant": self.backbone_variant,
                "num_classes": self.num_classes,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self._input_shape_val,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
