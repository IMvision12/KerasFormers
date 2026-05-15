import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils
from keras.src.utils.argument_validation import standardize_tuple

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import INCEPTIONV4_CONFIG, INCEPTIONV4_WEIGHTS
from .convert_inceptionv4_torch_to_keras import transfer_inceptionv4_weights


def conv_block(
    inputs,
    filters=None,
    kernel_size=1,
    strides=1,
    bn_momentum=0.9,
    bn_epsilon=1e-3,
    padding="valid",
    name="conv2d_block",
):
    """Conv -> BatchNorm -> ReLU with optional asymmetric ZeroPadding.

    Args:
        inputs: Input Keras tensor.
        filters: Number of output filters for the convolution.
        kernel_size: Int or 2-tuple kernel size. Scalars are expanded to
            ``(k, k)``.
        strides: Stride of the convolution.
        bn_momentum: Momentum for the BatchNormalization layer.
        bn_epsilon: Epsilon for the BatchNormalization layer.
        padding: Padding mode. ``None`` triggers explicit asymmetric
            zero-padding to match the timm reference layout.
        name: Name prefix used for the conv / bn / activation layers.

    Returns:
        Output tensor after Conv -> BN -> ReLU.
    """
    kernel_size = standardize_tuple(kernel_size, 2, "kernel_size")
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    x = inputs
    if padding is None:

        def calculate_padding(kernel_dim):
            pad_total = kernel_dim - 1
            pad_size = pad_total // 2
            pad_extra = (kernel_dim - 1) % 2
            return pad_size, pad_extra

        pad_h, extra_h = calculate_padding(kernel_size[0])
        pad_w, extra_w = calculate_padding(kernel_size[1])

        if strides > 1:
            padding_config = ((pad_h + extra_h, pad_h), (pad_w + extra_w, pad_w))
        else:
            padding_config = ((pad_h, pad_h), (pad_w, pad_w))

        x = layers.ZeroPadding2D(padding=padding_config, name=f"{name}_padding")(x)
        padding = "valid"

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=keras.config.image_data_format(),
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        name=f"{name}_bn",
    )(x)
    x = layers.Activation("relu", name=name)(x)
    return x


def stem_blocks(x, conv_block):
    """InceptionV4 stem: 3 conv layers before the Mixed blocks.

    Args:
        x: Input image tensor.
        conv_block: Callable that builds a Conv -> BN -> ReLU block.

    Returns:
        Tensor after the three stem convolutions.
    """
    x = conv_block(x, 32, kernel_size=3, strides=2, name="features_0")
    x = conv_block(x, 32, kernel_size=3, name="features_1")
    x = conv_block(x, 64, kernel_size=3, padding=None, name="features_2")
    return x


def mixed3a(x, conv_block, name="features_3"):
    """Mixed3a: MaxPool || strided 3x3 conv.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv -> BN -> ReLU block.
        name: Name prefix for layers in the block.

    Returns:
        Output tensor concatenating the maxpool and conv branches.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    maxpool = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)
    conv = conv_block(x, 96, kernel_size=3, strides=2, name=f"{name}_conv")
    return layers.Concatenate(axis=channels_axis, name=name)([maxpool, conv])


def mixed4a(x, conv_block, name="features_4"):
    """Mixed4a: parallel paths with 1x7 + 7x1 factorized convs.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv -> BN -> ReLU block.
        name: Name prefix for layers in the block.

    Returns:
        Output tensor concatenating the two branch outputs.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    branch0 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch0_0")
    branch0 = conv_block(
        branch0, 96, kernel_size=3, strides=1, name=f"{name}_branch0_1"
    )

    branch1 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1,
        64,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1",
    )
    branch1 = conv_block(
        branch1,
        64,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_2",
    )
    branch1 = conv_block(
        branch1, 96, kernel_size=3, strides=1, name=f"{name}_branch1_3"
    )

    return layers.Concatenate(axis=channels_axis, name=name)([branch0, branch1])


def mixed5a(x, conv_block, name="features_5"):
    """Mixed5a: strided 3x3 conv || maxpool.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv -> BN -> ReLU block.
        name: Name prefix for layers in the block.

    Returns:
        Spatially down-sampled output tensor.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    conv = conv_block(x, 192, kernel_size=3, strides=2, name=f"{name}_conv")
    maxpool = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)
    return layers.Concatenate(axis=channels_axis, name=name)([conv, maxpool])


def inception_a(x, conv_block, block_idx):
    """Inception-A block (4 parallel branches).

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv -> BN -> ReLU block.
        block_idx: Integer index used to assemble the ``features_{idx}`` name
            prefix.

    Returns:
        Output tensor concatenating the four branch outputs.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    name = f"features_{block_idx}"

    branch0 = conv_block(x, 96, kernel_size=1, strides=1, name=f"{name}_branch0")

    branch1 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1, 96, kernel_size=3, strides=1, padding=None, name=f"{name}_branch1_1"
    )

    branch2 = conv_block(x, 64, kernel_size=1, strides=1, name=f"{name}_branch2_0")
    branch2 = conv_block(
        branch2, 96, kernel_size=3, strides=1, padding=None, name=f"{name}_branch2_1"
    )
    branch2 = conv_block(
        branch2, 96, kernel_size=3, strides=1, padding=None, name=f"{name}_branch2_2"
    )

    branch3 = layers.AveragePooling2D(
        3, strides=1, padding="same", data_format=keras.config.image_data_format()
    )(x)
    branch3 = conv_block(
        branch3, 96, kernel_size=1, strides=1, name=f"{name}_branch3_1"
    )

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2, branch3]
    )


def reduction_a(x, conv_block, name="features_10"):
    """Reduction-A: spatial downsampling stage.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv -> BN -> ReLU block.
        name: Name prefix for layers in the block.

    Returns:
        Spatially down-sampled output tensor (3 concatenated branches).
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    branch0 = conv_block(x, 384, kernel_size=3, strides=2, name=f"{name}_branch0")

    branch1 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1, 224, kernel_size=3, strides=1, padding=None, name=f"{name}_branch1_1"
    )
    branch1 = conv_block(
        branch1, 256, kernel_size=3, strides=2, name=f"{name}_branch1_2"
    )

    branch2 = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2]
    )


def inception_b(x, conv_block, block_idx):
    """Inception-B block with 1x7 and 7x1 factorized convolutions.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv -> BN -> ReLU block.
        block_idx: Integer index used to assemble the ``features_{idx}`` name
            prefix.

    Returns:
        Output tensor concatenating the four branch outputs.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    name = f"features_{block_idx}"

    branch0 = conv_block(x, 384, kernel_size=1, strides=1, name=f"{name}_branch0")

    branch1 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1,
        224,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1",
    )
    branch1 = conv_block(
        branch1,
        256,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_2",
    )

    branch2 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch2_0")
    branch2 = conv_block(
        branch2,
        192,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_1",
    )
    branch2 = conv_block(
        branch2,
        224,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch2_2",
    )
    branch2 = conv_block(
        branch2,
        224,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_3",
    )
    branch2 = conv_block(
        branch2,
        256,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch2_4",
    )

    branch3 = layers.AveragePooling2D(
        3, strides=1, padding="same", data_format=keras.config.image_data_format()
    )(x)
    branch3 = conv_block(
        branch3, 128, kernel_size=1, strides=1, name=f"{name}_branch3_1"
    )

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2, branch3]
    )


def reduction_b(x, conv_block, name="features_18"):
    """Reduction-B: spatial downsampling stage.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv -> BN -> ReLU block.
        name: Name prefix for layers in the block.

    Returns:
        Spatially down-sampled output tensor (3 concatenated branches).
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    branch0 = conv_block(x, 192, kernel_size=1, strides=1, name=f"{name}_branch0_0")
    branch0 = conv_block(
        branch0, 192, kernel_size=3, strides=2, name=f"{name}_branch0_1"
    )

    branch1 = conv_block(x, 256, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1 = conv_block(
        branch1,
        256,
        kernel_size=(1, 7),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1",
    )
    branch1 = conv_block(
        branch1,
        320,
        kernel_size=(7, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_2",
    )
    branch1 = conv_block(
        branch1, 320, kernel_size=3, strides=2, name=f"{name}_branch1_3"
    )

    branch2 = layers.MaxPooling2D(
        3, strides=2, data_format=keras.config.image_data_format()
    )(x)

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2]
    )


def inception_c(x, conv_block, block_idx):
    """Inception-C block with split 1x3 and 3x1 parallel convolutions.

    Args:
        x: Input Keras tensor.
        conv_block: Callable that builds a Conv -> BN -> ReLU block.
        block_idx: Integer index used to assemble the ``features_{idx}`` name
            prefix.

    Returns:
        Output tensor concatenating the four expanded branch outputs.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1
    name = f"features_{block_idx}"

    branch0 = conv_block(x, 256, kernel_size=1, strides=1, name=f"{name}_branch0")

    branch1 = conv_block(x, 384, kernel_size=1, strides=1, name=f"{name}_branch1_0")
    branch1_1a = conv_block(
        branch1,
        256,
        kernel_size=(1, 3),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1a",
    )
    branch1_1b = conv_block(
        branch1,
        256,
        kernel_size=(3, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch1_1b",
    )
    branch1 = layers.Concatenate(axis=channels_axis)([branch1_1a, branch1_1b])

    branch2 = conv_block(x, 384, kernel_size=1, strides=1, name=f"{name}_branch2_0")
    branch2 = conv_block(
        branch2,
        448,
        kernel_size=(3, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_1",
    )
    branch2 = conv_block(
        branch2,
        512,
        kernel_size=(1, 3),
        strides=1,
        padding=None,
        name=f"{name}_branch2_2",
    )
    branch2_3a = conv_block(
        branch2,
        256,
        kernel_size=(1, 3),
        strides=1,
        padding=None,
        name=f"{name}_branch2_3a",
    )
    branch2_3b = conv_block(
        branch2,
        256,
        kernel_size=(3, 1),
        strides=1,
        padding=None,
        name=f"{name}_branch2_3b",
    )
    branch2 = layers.Concatenate(axis=channels_axis)([branch2_3a, branch2_3b])

    branch3 = layers.AveragePooling2D(
        3, strides=1, padding="same", data_format=keras.config.image_data_format()
    )(x)
    branch3 = conv_block(
        branch3, 256, kernel_size=1, strides=1, name=f"{name}_branch3_1"
    )

    return layers.Concatenate(axis=channels_axis, name=name)(
        [branch0, branch1, branch2, branch3]
    )


def inceptionv4_backbone_feature(inputs, *, data_format, return_stages=False):
    """InceptionV4 full backbone, returns the final stage feature map.

    Args:
        inputs: Input image tensor.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_stages: If True, return a list of per-stage feature maps
            taken at natural downsample boundaries (after Mixed3a, after
            Inception-A group, after Inception-B group, after Inception-C
            group). If False (default), return only the final stage map.

    Returns:
        Final stage feature tensor (after the last Inception-C block), or
        a list of per-stage feature maps when ``return_stages=True``.
    """
    stages = []
    x = stem_blocks(inputs, conv_block)

    x = mixed3a(x, conv_block)
    stages.append(x)  # stage 1: after Mixed3a (stride 4)

    x = mixed4a(x, conv_block)
    x = mixed5a(x, conv_block)

    for i in range(4):
        x = inception_a(x, conv_block, block_idx=6 + i)
    stages.append(x)  # stage 2: after Inception-A group (stride 8)

    x = reduction_a(x, conv_block)
    for i in range(7):
        x = inception_b(x, conv_block, block_idx=11 + i)
    stages.append(x)  # stage 3: after Inception-B group (stride 16)

    x = reduction_b(x, conv_block)
    for i in range(3):
        x = inception_c(x, conv_block, block_idx=19 + i)
    stages.append(x)  # stage 4: after Inception-C group (stride 32)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class InceptionV4Model(BaseModel):
    """InceptionV4 trunk returning the final stage feature map.

    Output shape: ``(B, H, W, C)`` — unpooled, head-free.
    :class:`InceptionV4Classify` composes this model and applies GAP +
    Dense head to produce logits.
    """

    KMODELS_CONFIG = INCEPTIONV4_CONFIG
    KMODELS_WEIGHTS = INCEPTIONV4_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = InceptionV4Classify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_inceptionv4_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        as_backbone=False,
        name="InceptionV4Model",
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
        x = inceptionv4_backbone_feature(
            x, data_format=data_format, return_stages=as_backbone
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
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
class InceptionV4Classify(BaseModel):
    """InceptionV4 classifier (timm-ported).

    Wraps an :class:`InceptionV4Model` backbone and applies a GAP + Dense
    head to produce class logits.

    Reference:
    - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) (AAAI 2017)

    Construction:

    >>> InceptionV4Classify.from_weights("inception_v4_tf_in1k")
    >>> InceptionV4Classify.from_weights("timm:timm/inception_v4.tf_in1k")
    """

    KMODELS_CONFIG = INCEPTIONV4_CONFIG
    KMODELS_WEIGHTS = INCEPTIONV4_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_inceptionv4_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="InceptionV4Classify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = InceptionV4Model(
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
