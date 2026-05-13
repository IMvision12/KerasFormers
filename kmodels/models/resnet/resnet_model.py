from typing import Optional

import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import RESNET_CONFIG, RESNET_WEIGHTS
from .convert_resnet_torch_to_keras import transfer_resnet_weights


def conv_block(
    x: layers.Layer,
    filters: int,
    kernel_size: int,
    channels_axis,
    data_format,
    strides: int = 1,
    use_relu: bool = True,
    groups: int = 1,
    group_width: Optional[int] = None,
    name: Optional[str] = None,
    bn_name: Optional[str] = None,
) -> layers.Layer:
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
        group_width: Width per group (used if groups > 1).
        name: Optional name for the convolution layer.
        bn_name: Optional name for the batch normalization layer.

    Returns:
       Output tensor for the block.
    """
    pad_h = pad_w = kernel_size // 2

    if strides > 1:
        x = layers.ZeroPadding2D(data_format=data_format, padding=(pad_h, pad_w))(x)
        padding = "valid"
    else:
        padding = "same"

    if groups > 1:
        assert filters % groups == 0, (
            f"Filters ({filters}) must be divisible by groups ({groups})"
        )
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            groups=groups,
            kernel_initializer="he_normal",
            data_format=data_format,
            name=name,
        )(x)
    else:
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer="he_normal",
            data_format=data_format,
            name=name,
        )(x)

    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=bn_name
    )(x)

    if use_relu:
        x = layers.ReLU()(x)
    return x


def squeeze_excitation_block(
    x: layers.Layer, data_format, reduction_ratio: int = 16, name: Optional[str] = None
) -> layers.Layer:
    """
    Squeeze-and-Excitation block that properly handles both channels_first and channels_last formats.

    Args:
        x: Input tensor
        data_format: String, either 'channels_first' or 'channels_last'
        reduction_ratio: Integer, reduction ratio for the bottleneck
        name: String, optional name prefix for layers

    Returns:
        Tensor with same shape as input after applying SE attention
    """
    if data_format == "channels_first":
        channel_axis = 1
        filters = x.shape[channel_axis]
    else:
        channel_axis = -1
        filters = x.shape[channel_axis]

    se = layers.GlobalAveragePooling2D(data_format=data_format)(x)

    if data_format == "channels_first":
        se = layers.Reshape((filters, 1, 1))(se)
    else:
        se = layers.Reshape((1, 1, filters))(se)

    reduced_filters = max(1, filters // reduction_ratio)
    se = layers.Reshape((filters,))(se)
    se = layers.Dense(
        reduced_filters,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_dense1" if name else None,
    )(se)
    se = layers.Dense(
        filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_dense2" if name else None,
    )(se)

    if data_format == "channels_first":
        se = layers.Reshape((filters, 1, 1))(se)
    else:
        se = layers.Reshape((1, 1, filters))(se)

    return layers.Multiply(name=f"{name}_scale" if name else None)([x, se])


def bottleneck_block(
    x: layers.Layer,
    filters: int,
    channels_axis,
    data_format,
    strides: int = 1,
    downsample: bool = False,
    senet: bool = False,
    block_name: Optional[str] = None,
) -> layers.Layer:
    """Bottleneck ResNet block.

    Args:
        x: Input Keras layer.
        filters: Number of filters for the bottleneck layers.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        strides: Stride for the main convolution layer.
        downsample: Whether to downsample the input.
        senet: Whether to apply SE block.
        block_name: Optional name for layers in the block.

    Returns:
        Output tensor for the block.
    """
    residual = x
    expansion = 4

    x = conv_block(
        x,
        filters,
        kernel_size=1,
        strides=1,
        name=f"{block_name}_conv1",
        bn_name=f"{block_name}_batchnorm1",
        channels_axis=channels_axis,
        data_format=data_format,
    )
    x = conv_block(
        x,
        filters,
        kernel_size=3,
        strides=strides,
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


def _resnet_features(
    inputs,
    block_fn,
    block_repeats,
    filters,
    channels_axis,
    data_format,
    groups,
    senet,
    width_factor,
):
    """Build the ResNet stem + stages, returning a list of feature maps.

    Returns ``[stem, stage1, stage2, stage3, stage4]`` (5 tensors). Shared
    by :class:`ResNet` (which pools + classifies the last one) and
    :class:`ResNetBackbone` (which exposes the full list).
    """
    features = []

    x = conv_block(
        inputs,
        filters[0],
        kernel_size=7,
        strides=2,
        name="conv1",
        bn_name="batchnorm1",
        channels_axis=channels_axis,
        data_format=data_format,
    )
    x = layers.ZeroPadding2D(data_format=data_format, padding=(1, 1))(x)
    x = layers.MaxPooling2D(
        data_format=data_format, pool_size=3, strides=2, padding="valid"
    )(x)
    features.append(x)

    common_args = {
        "channels_axis": channels_axis,
        "data_format": data_format,
        "senet": senet,
    }
    if isinstance(block_fn, dict):
        if block_fn.get("module") == "kmodels.models.resnext.resnext_model":
            common_args.update({"groups": groups, "width_factor": width_factor})
    elif hasattr(block_fn, "__module__") and "resnext" in block_fn.__module__:
        common_args.update({"groups": groups, "width_factor": width_factor})

    for i, num_blocks in enumerate(block_repeats):
        for j in range(num_blocks):
            common_args["block_name"] = f"resnet_layer{i + 1}_{j}"
            if j == 0 and i > 0:
                x = block_fn(x, filters[i], strides=2, downsample=True, **common_args)
            else:
                x = block_fn(x, filters[i], **common_args)
        features.append(x)

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class ResNet(BaseModel):
    """
    Instantiates the ResNet architecture with support for ResNeXt and SE-ResNet/SE-ResNeXt configurations.

    Construction:

    >>> ResNet.from_weights("resnet50_a1_in1k")               # kmodels release
    >>> ResNet.from_weights("timm:user/my-resnet50-finetune", variant="resnet50_a1_in1k")
    >>> ResNet.from_weights("resnet50_a1_in1k", as_backbone=True, include_top=False)

    Reference:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2016)

    Args:
        block_fn: Callable, the block function to use for residual blocks. Should accept parameters:
            (x, filters, strides=1, downsample=False, block_name=None)
        block_repeats: List of integers, number of blocks to repeat at each stage.
        filters: List of integers, number of filters for each stage.
        groups: Integer, number of groups for group convolutions in ResNeXt blocks.
            Default is `32`.
        senet: Boolean, whether to include Squeeze-and-Excitation (SE) blocks for improved feature recalibration.
            Default is `False`.
        width_factor: Integer, scaling factor for the width of ResNeXt blocks.
            Default is `2`.
        include_top: Boolean, whether to include the fully-connected classification
            layer at the top. Defaults to `True`.
        as_backbone: Boolean, whether to output intermediate features for use as a
            backbone network. When True, returns a list of feature maps at different
            stages. Defaults to `False`.
        include_normalization: Boolean, whether to include normalization layers at the start
            of the network. When True, input images should be in uint8 format with values
            in [0, 255]. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to use. Must be one of:
            'imagenet' (default), 'inception', 'dpn', 'clip', 'zero_to_one', or
            'minus_one_to_one'. Only used when include_normalization=True.
        weights: String, specifying the path to pretrained weights or one of the
            available options in `keras-vision`.
        input_tensor: Optional Keras tensor to use as the model's input. If not provided,
            a new input tensor is created based on `input_shape`.
        input_shape: Optional tuple specifying the shape of the input data. If not
            specified, defaults to `(224, 224, 3)`.
        pooling: Optional pooling mode for feature extraction when `include_top=False`:
            - `None` (default): the output is the 4D tensor from the last convolutional block.
            - `"avg"`: global average pooling is applied, and the output is a 2D tensor.
            - `"max"`: global max pooling is applied, and the output is a 2D tensor.
        num_classes: Integer, the number of output classes for classification. Defaults to `1000`.
            Only applicable if `include_top=True`.
        classifier_activation: String or callable, activation function for the
            classifier layer. Set to `None` to return logits.
            Defaults to `"linear"`.
        name: String, the name of the model. Defaults to `"ResNet"`.

    Returns:
        A Keras `Model` instance.
    """

    KMODELS_CONFIG = RESNET_CONFIG
    KMODELS_WEIGHTS = RESNET_WEIGHTS
    HF_MODEL_TYPE = None  # timm-ported; no HF transformers passthrough.

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_resnet_weights(keras_model, state_dict)

    def __init__(
        self,
        block_fn=bottleneck_block,
        block_repeats=[2, 2, 2, 2],
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
        name="ResNet",
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
        features = _resnet_features(
            x,
            block_fn=block_fn,
            block_repeats=block_repeats,
            filters=filters,
            channels_axis=channels_axis,
            data_format=data_format,
            groups=groups,
            senet=senet,
            width_factor=width_factor,
        )
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.Dense(
            num_classes,
            activation=classifier_activation,
            kernel_initializer="zeros",
            name="predictions",
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

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

    def get_config(self):
        config = super().get_config()

        if hasattr(self.block_fn, "__module__"):
            block_fn_config = {
                "class_name": "function",
                "config": self.block_fn.__name__,
                "module": self.block_fn.__module__,
                "registered_name": "function",
            }
        else:
            block_fn_config = {
                "class_name": "function",
                "config": "bottleneck_block",
                "module": "kv.models.resnet.resnet_model",
                "registered_name": "function",
            }

        config.update(
            {
                "block_fn": block_fn_config,
                "block_repeats": self.block_repeats,
                "filters": self.filters,
                "groups": self.groups,
                "senet": self.senet,
                "width_factor": self.width_factor,
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
        if isinstance(config["block_fn"], dict):
            block_fn_name = config["block_fn"]["config"]
            module_path = config["block_fn"]["module"]

            if module_path == "kmodels.models.resnet.resnet_model":
                if block_fn_name == "bottleneck_block":
                    config["block_fn"] = bottleneck_block
            elif module_path == "kmodels.models.resnext.resnext_model":
                from kmodels.models.resnext.resnext_model import resnext_block

                if block_fn_name == "resnext_block":
                    config["block_fn"] = resnext_block
            else:
                raise ValueError(
                    f"Unknown block function: {block_fn_name} from module {module_path}"
                )

        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class ResNetBackbone(BaseModel):
    """ResNet feature extractor (no classifier head).

    Returns a list of five feature maps: ``[stem, stage1, stage2,
    stage3, stage4]``, sized at strides ``[4, 4, 8, 16, 32]`` of the
    input. Use as a frozen / fine-tunable backbone in detection,
    segmentation, or other downstream tasks.

    Construction:

    >>> ResNetBackbone.from_weights("resnet50_a1_in1k")
    >>> ResNetBackbone.from_weights("timm:timm/resnet50.a1_in1k")
    """

    KMODELS_CONFIG = RESNET_CONFIG
    KMODELS_WEIGHTS = RESNET_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        """Classifier class that owns the kmodels release ``.weights.h5``.

        Subclasses (e.g. :class:`ResNeXtBackbone`) override this to point
        at their matching classifier model.
        """
        return ResNet

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        """Build backbone and warm-start from the matching classifier release.

        The kmodels release files are saved by the classifier class
        (which includes the Dense head); we materialize a temporary
        classifier, load its weights, then copy the matching encoder
        layers here by stable path suffix. The classifier weights are
        discarded.
        """
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = cls._release_warm_start_cls().from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_resnet_weights(keras_model, state_dict)

    def __init__(
        self,
        block_fn=bottleneck_block,
        block_repeats=[2, 2, 2, 2],
        filters=[64, 128, 256, 512],
        groups=32,
        senet=False,
        width_factor=2,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="ResNetBackbone",
        **kwargs,
    ):
        # ResNet classifier-only kwargs that arrive via KMODELS_CONFIG / a
        # ResNet config dict — accept-and-drop so the shared config is usable.
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
        features = _resnet_features(
            x,
            block_fn=block_fn,
            block_repeats=block_repeats,
            filters=filters,
            channels_axis=channels_axis,
            data_format=data_format,
            groups=groups,
            senet=senet,
            width_factor=width_factor,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.block_fn = block_fn
        self.block_repeats = block_repeats
        self.filters = filters
        self.groups = groups
        self.senet = senet
        self.width_factor = width_factor
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        if hasattr(self.block_fn, "__module__"):
            block_fn_config = {
                "class_name": "function",
                "config": self.block_fn.__name__,
                "module": self.block_fn.__module__,
                "registered_name": "function",
            }
        else:
            block_fn_config = {
                "class_name": "function",
                "config": "bottleneck_block",
                "module": "kmodels.models.resnet.resnet_model",
                "registered_name": "function",
            }
        config.update(
            {
                "block_fn": block_fn_config,
                "block_repeats": self.block_repeats,
                "filters": self.filters,
                "groups": self.groups,
                "senet": self.senet,
                "width_factor": self.width_factor,
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
        if isinstance(config.get("block_fn"), dict):
            block_fn_name = config["block_fn"]["config"]
            module_path = config["block_fn"]["module"]
            if module_path == "kmodels.models.resnet.resnet_model":
                if block_fn_name == "bottleneck_block":
                    config["block_fn"] = bottleneck_block
            elif module_path == "kmodels.models.resnext.resnext_model":
                from kmodels.models.resnext.resnext_model import resnext_block

                if block_fn_name == "resnext_block":
                    config["block_fn"] = resnext_block
            else:
                raise ValueError(
                    f"Unknown block function: {block_fn_name} from module {module_path}"
                )
        return cls(**config)
