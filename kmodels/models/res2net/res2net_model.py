import keras
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import RES2NET_CONFIG, RES2NET_WEIGHTS
from .convert_res2net_torch_to_keras import transfer_res2net_weights


def conv_block(
    x,
    filters,
    kernel_size,
    channels_axis,
    data_format,
    strides=1,
    use_relu=True,
    groups=1,
    name=None,
    bn_name=None,
):
    """Applies a convolution block with optional grouped convolutions.

    Args:
        x: Input Keras layer.
        filters: Number of output filters for the convolution.
        kernel_size: Size of the convolution kernel.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        strides: Stride of the convolution.
        use_relu: Whether to apply ReLU activation after convolution.
        groups: Number of groups for grouped convolution.
        name: Optional name for the convolution layer.
        bn_name: Optional name for the batch normalization layer.

    Returns:
       Output tensor for the block.
    """
    if strides > 1:
        pad_h = pad_w = kernel_size // 2
        x = layers.ZeroPadding2D(padding=(pad_h, pad_w), data_format=data_format)(x)
        padding_mode = "valid"
    else:
        padding_mode = "same"

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding_mode,
        use_bias=False,
        groups=groups,
        data_format=data_format,
        name=name,
    )(x)

    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        name=bn_name,
    )(x)

    if use_relu:
        x = layers.ReLU()(x)
    return x


def bottle2neck_block(
    x,
    filters,
    block_name,
    data_format,
    stride=1,
    downsample=False,
    cardinality=1,
    base_width=26,
    scale=4,
):
    """Res2Net/ResNeSt Bottle2neck block with multi-scale features.

    Args:
        x: Input Keras layer.
        filters: Number of filters for the bottleneck layers.
        block_name: Name prefix for layers in the block.
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        stride: Stride for the 3x3 convolution layers.
        downsample: Whether to downsample the input.
        cardinality: Number of groups for grouped convolutions.
        base_width: Base width of the block, controls channel scaling.
        scale: Scale factor that determines number of feature scales.

    Returns:
        Output tensor for the block.

    Notes:
        The block implements multi-scale feature processing by:
        1. Initial 1x1 conv to expand channels
        2. Split features into multiple scales
        3. Hierarchical residual-like connections between scales
        4. Optional average pooling for the last scale
        5. Concatenate all scales and reduce with 1x1 conv

        The expansion factor is fixed at 4, similar to standard ResNet bottleneck blocks.
    """
    channels_axis = -1 if data_format == "channels_last" else 1
    expansion = 4
    is_first = stride > 1 or downsample
    width = int(filters * (base_width / 64.0)) * cardinality
    outplanes = filters * expansion

    identity = x

    x = conv_block(
        x,
        width * scale,
        kernel_size=1,
        channels_axis=channels_axis,
        data_format=data_format,
        name=f"{block_name}_conv_1",
        bn_name=f"{block_name}_batchnorm_1",
    )

    x_splits = ops.split(x, scale, axis=channels_axis)
    spouts = []

    for i in range(scale - 1):
        if i == 0 or is_first:
            sp = x_splits[i]
        else:
            sp = layers.Add()([spouts[-1], x_splits[i]])

        sp = conv_block(
            sp,
            width,
            kernel_size=3,
            channels_axis=channels_axis,
            data_format=data_format,
            strides=stride if is_first else 1,
            groups=cardinality,
            name=f"{block_name}_conv_s_{i}",
            bn_name=f"{block_name}_batchnorm_s_{i}",
        )
        spouts.append(sp)

    if scale > 1:
        if is_first:
            last = layers.ZeroPadding2D(
                padding=((1, 1), (1, 1)), data_format=data_format
            )(x_splits[-1])
            last = layers.AveragePooling2D(
                pool_size=3,
                strides=stride,
                padding="valid",
                data_format=data_format,
            )(last)
        else:
            last = x_splits[-1]
        spouts.append(last)

    out = layers.Concatenate(axis=channels_axis)(spouts)

    out = conv_block(
        out,
        outplanes,
        kernel_size=1,
        channels_axis=channels_axis,
        data_format=data_format,
        use_relu=False,
        name=f"{block_name}_conv_3",
        bn_name=f"{block_name}_batchnorm_3",
    )

    if downsample:
        identity = conv_block(
            identity,
            outplanes,
            kernel_size=1,
            channels_axis=channels_axis,
            data_format=data_format,
            strides=stride,
            use_relu=False,
            name=f"{block_name}_downsample_0",
            bn_name=f"{block_name}_downsample_1",
        )

    out = layers.Add()([identity, out])
    out = layers.ReLU()(out)

    return out


def res2net_backbone_feature(
    inputs,
    depth,
    base_width,
    scale,
    cardinality,
    channels_axis,
    data_format,
):
    """Res2Net stem + stages, returning a list of feature maps.

    Shared by :class:`Res2Net` (which pools + classifies) and
    :class:`Res2NetBackbone` (which exposes the full list).
    """
    features = []
    x = layers.ZeroPadding2D(padding=3, data_format=data_format)(inputs)
    x = layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name="bn1"
    )(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(data_format=data_format, padding=(1, 1))(x)
    x = layers.MaxPooling2D(
        pool_size=3, strides=2, padding="valid", data_format=data_format
    )(x)
    features.append(x)

    filters = [64, 128, 256, 512]
    for i, (blocks, filter_size) in enumerate(zip(depth, filters)):
        stride = 1 if i == 0 else 2
        x = bottle2neck_block(
            x,
            filter_size,
            f"layer{i + 1}_0",
            stride=stride,
            downsample=True,
            base_width=base_width,
            cardinality=cardinality,
            scale=scale,
            data_format=data_format,
        )
        for j in range(1, blocks):
            x = bottle2neck_block(
                x,
                filter_size,
                f"layer{i + 1}_{j}",
                base_width=base_width,
                cardinality=cardinality,
                scale=scale,
                data_format=data_format,
            )
        features.append(x)
    return features


@keras.saving.register_keras_serializable(package="kmodels")
class Res2NetClassify(BaseModel):
    """Res2Net classifier with multi-scale residual blocks.

    Reference:
    - [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169) (TPAMI 2019)

    Construction:

    >>> Res2NetClassify.from_weights("res2net50_26w_4s_in1k")
    >>> Res2NetClassify.from_weights("timm:timm/res2net50_26w_4s.in1k")

    Use :class:`Res2NetBackbone` for the per-stage feature maps.

    Args:
        depth: List of ints, number of blocks per stage.
        base_width: Int, base channel width per scale. Default ``26``.
        scale: Int, number of scales per Res2Net block. Default ``4``.
        cardinality: Int, group count for grouped convolution. Default ``1``.
        include_normalization: Bool, whether to prepend an
            :class:`ImageNormalizationLayer`. Default ``True``.
        normalization_mode: One of ``"imagenet"``, ``"inception"``, ``"dpn"``,
            ``"clip"``, ``"zero_to_one"``, ``"minus_one_to_one"``. Default
            ``"imagenet"``.
        input_shape: Optional ``(H, W, C)``. Default ``(224, 224, 3)``.
        input_tensor: Optional pre-existing Keras input tensor.
        num_classes: Int, number of output classes. Default ``1000``.
        classifier_activation: Activation for the head. ``None`` returns
            logits. Default ``"linear"``.
        name: Model name. Default ``"Res2NetClassify"``.

    Returns:
        A Keras :class:`Model` instance.
    """

    KMODELS_CONFIG = RES2NET_CONFIG
    KMODELS_WEIGHTS = RES2NET_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_res2net_weights(keras_model, state_dict)

    def __init__(
        self,
        depth=(3, 4, 6, 3),
        base_width=26,
        scale=4,
        cardinality=1,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        num_classes=1000,
        classifier_activation="linear",
        name="Res2NetClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=224,
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
        features = res2net_backbone_feature(
            x,
            depth=depth,
            base_width=base_width,
            scale=scale,
            cardinality=cardinality,
            channels_axis=channels_axis,
            data_format=data_format,
        )
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depth = depth
        self.base_width = base_width
        self.scale = scale
        self.cardinality = cardinality
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "base_width": self.base_width,
                "scale": self.scale,
                "cardinality": self.cardinality,
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


@keras.saving.register_keras_serializable(package="kmodels")
class Res2NetModel(BaseModel):
    """Res2Net trunk returning the final stage feature map ``(B, H, W, C)``."""

    KMODELS_CONFIG = RES2NET_CONFIG
    KMODELS_WEIGHTS = RES2NET_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = Res2NetClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_res2net_weights(keras_model, state_dict)

    def __init__(
        self,
        depth=(3, 4, 6, 3),
        base_width=26,
        scale=4,
        cardinality=1,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        name="Res2NetModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=224,
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
        features = res2net_backbone_feature(
            x,
            depth=depth,
            base_width=base_width,
            scale=scale,
            cardinality=cardinality,
            channels_axis=channels_axis,
            data_format=data_format,
        )

        super().__init__(inputs=img_input, outputs=features[-1], name=name, **kwargs)

        self.depth = depth
        self.base_width = base_width
        self.scale = scale
        self.cardinality = cardinality
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "base_width": self.base_width,
                "scale": self.scale,
                "cardinality": self.cardinality,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class Res2NetBackbone(BaseModel):
    """Res2Net feature extractor (no classifier head).

    Returns a list ``[stem, stage1, stage2, stage3, stage4]`` of feature
    maps. Use as a backbone for detection / segmentation downstream.
    """

    KMODELS_CONFIG = RES2NET_CONFIG
    KMODELS_WEIGHTS = RES2NET_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = Res2NetClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_res2net_weights(keras_model, state_dict)

    def __init__(
        self,
        depth=(3, 4, 6, 3),
        base_width=26,
        scale=4,
        cardinality=1,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        name="Res2NetBackbone",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=224,
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
        features = res2net_backbone_feature(
            x,
            depth=depth,
            base_width=base_width,
            scale=scale,
            cardinality=cardinality,
            channels_axis=channels_axis,
            data_format=data_format,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.depth = depth
        self.base_width = base_width
        self.scale = scale
        self.cardinality = cardinality
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "base_width": self.base_width,
                "scale": self.scale,
                "cardinality": self.cardinality,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
