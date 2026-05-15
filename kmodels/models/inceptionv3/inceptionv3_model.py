import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils
from keras.src.utils.argument_validation import standardize_tuple

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import INCEPTIONV3_MODEL_CONFIG, INCEPTIONV3_WEIGHT_CONFIG
from .convert_inceptionv3_torch_to_keras import transfer_inceptionv3_weights


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
        name=f"{name}_conv2d",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        name=f"{name}_batchnorm",
    )(x)
    x = layers.Activation("relu", name=name)(x)
    return x


def inception_blocka(inputs, pool_channels, name="inception_block_a"):
    """Inception block type A (1x1, 5x5, double-3x3, avg-pool).

    Args:
        inputs: Input Keras tensor.
        pool_channels: Number of filters for the average-pool 1x1 projection
            branch (differs across Mixed_5b/5c/5d).
        name: Name prefix for layers in the block.

    Returns:
        Output tensor formed by concatenating the four branch outputs along
        the channel axis.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch1x1 = conv_block(inputs, 64, 1, name=f"{name}_branch1x1")

    branch5x5 = conv_block(inputs, 48, 1, name=f"{name}_branch5x5_1")
    branch5x5 = conv_block(branch5x5, 64, 5, padding=None, name=f"{name}_branch5x5_2")

    branch3x3dbl = conv_block(inputs, 64, 1, name=f"{name}_branch3x3dbl_1")
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, padding=None, name=f"{name}_branch3x3dbl_3"
    )

    branch_pool = layers.ZeroPadding2D(
        data_format=keras.config.image_data_format(), padding=1
    )(inputs)
    branch_pool = layers.AveragePooling2D(
        pool_size=3,
        strides=1,
        data_format=keras.config.image_data_format(),
    )(branch_pool)
    branch_pool = conv_block(
        branch_pool,
        pool_channels,
        name=f"{name}_branch_pool",
    )

    return layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool]
    )


def inception_blockb(inputs, name="inception_block_b"):
    """Inception block type B (strided 3x3, double-3x3 strided, maxpool).

    Args:
        inputs: Input Keras tensor.
        name: Name prefix for layers in the block.

    Returns:
        Spatially down-sampled output tensor (3 concatenated branches).
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch3x3 = conv_block(inputs, 384, 3, 2, name=f"{name}_branch3x3")

    branch3x3dbl = conv_block(inputs, 64, 1, name=f"{name}_branch3x3dbl_1")
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, strides=2, name=f"{name}_branch3x3dbl_3"
    )

    branch_pool = layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        data_format=keras.config.image_data_format(),
        name=f"{name}_branch_pool",
    )(inputs)

    return layers.Concatenate(axis=channels_axis)(
        [branch3x3, branch3x3dbl, branch_pool]
    )


def inception_blockc(inputs, branch7x7_channels, name="inception_block_c"):
    """Inception block type C (factorized 7x7 = 1x7 + 7x1).

    Args:
        inputs: Input Keras tensor.
        branch7x7_channels: Inner channel count for the 7x7 / 7x7-double
            branches (differs across Mixed_6b/6c/6d/6e).
        name: Name prefix for layers in the block.

    Returns:
        Output tensor concatenating the four branch outputs along the
        channel axis.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    c7 = branch7x7_channels

    branch1x1 = conv_block(inputs, 192, 1, name=f"{name}_branch1x1")

    branch7x7 = conv_block(inputs, c7, 1, name=f"{name}_branch7x7_1")
    branch7x7 = conv_block(
        branch7x7, c7, (1, 7), padding=None, name=f"{name}_branch7x7_2"
    )
    branch7x7 = conv_block(
        branch7x7, 192, (7, 1), padding=None, name=f"{name}_branch7x7_3"
    )

    branch7x7dbl = conv_block(inputs, c7, 1, name=f"{name}_branch7x7dbl_1")
    branch7x7dbl = conv_block(
        branch7x7dbl, c7, (7, 1), padding=None, name=f"{name}_branch7x7dbl_2"
    )
    branch7x7dbl = conv_block(
        branch7x7dbl, c7, (1, 7), padding=None, name=f"{name}_branch7x7dbl_3"
    )
    branch7x7dbl = conv_block(
        branch7x7dbl, c7, (7, 1), padding=None, name=f"{name}_branch7x7dbl_4"
    )
    branch7x7dbl = conv_block(
        branch7x7dbl, 192, (1, 7), padding=None, name=f"{name}_branch7x7dbl_5"
    )

    branch_pool = layers.ZeroPadding2D(
        data_format=keras.config.image_data_format(), padding=1
    )(inputs)
    branch_pool = layers.AveragePooling2D(
        pool_size=3, strides=1, data_format=keras.config.image_data_format()
    )(branch_pool)
    branch_pool = conv_block(branch_pool, 192, 1, name=f"{name}_branch_pool")

    return layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool]
    )


def inception_blockd(inputs, name="inception_block_d"):
    """Inception block type D (reduction).

    Args:
        inputs: Input Keras tensor.
        name: Name prefix for layers in the block.

    Returns:
        Spatially down-sampled output tensor (3 concatenated branches).
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch3x3 = conv_block(inputs, 192, 1, name=f"{name}_branch3x3_1")
    branch3x3 = conv_block(branch3x3, 320, 3, strides=2, name=f"{name}_branch3x3_2")

    branch7x7x3 = conv_block(inputs, 192, 1, name=f"{name}_branch7x7x3_1")
    branch7x7x3 = conv_block(
        branch7x7x3, 192, (1, 7), padding=None, name=f"{name}_branch7x7x3_2"
    )
    branch7x7x3 = conv_block(
        branch7x7x3, 192, (7, 1), padding=None, name=f"{name}_branch7x7x3_3"
    )
    branch7x7x3 = conv_block(
        branch7x7x3, 192, 3, strides=2, name=f"{name}_branch7x7x3_4"
    )

    branch_pool = layers.MaxPooling2D(
        data_format=keras.config.image_data_format(), pool_size=3, strides=2
    )(inputs)

    return layers.Concatenate(axis=channels_axis)([branch3x3, branch7x7x3, branch_pool])


def inception_blocke(inputs, name="inception_block_e"):
    """Inception block type E (parallel factorized 3x3 = 1x3 || 3x1).

    Args:
        inputs: Input Keras tensor.
        name: Name prefix for layers in the block.

    Returns:
        Output tensor concatenating the four expanded branch outputs along
        the channel axis.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch1x1 = conv_block(inputs, 320, 1, name=f"{name}_branch1x1")

    branch3x3 = conv_block(inputs, 384, 1, name=f"{name}_branch3x3_1")
    branch3x3_a = conv_block(
        branch3x3,
        filters=384,
        kernel_size=(1, 3),
        padding=None,
        name=f"{name}_branch3x3_2a",
    )
    branch3x3_b = conv_block(
        branch3x3,
        filters=384,
        kernel_size=(3, 1),
        padding=None,
        name=f"{name}_branch3x3_2b",
    )
    branch3x3 = layers.Concatenate(axis=channels_axis)([branch3x3_a, branch3x3_b])

    branch3x3dbl = conv_block(inputs, 448, 1, name=f"{name}_branch3x3dbl_1")
    branch3x3dbl = conv_block(
        branch3x3dbl, 384, 3, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl_a = conv_block(
        branch3x3dbl,
        filters=384,
        kernel_size=(1, 3),
        padding=None,
        name=f"{name}_branch3x3dbl_3a",
    )
    branch3x3dbl_b = conv_block(
        branch3x3dbl,
        filters=384,
        kernel_size=(3, 1),
        padding=None,
        name=f"{name}_branch3x3dbl_3b",
    )
    branch3x3dbl = layers.Concatenate(axis=channels_axis)(
        [branch3x3dbl_a, branch3x3dbl_b]
    )

    branch_pool = layers.ZeroPadding2D(
        data_format=keras.config.image_data_format(), padding=1
    )(inputs)
    branch_pool = layers.AveragePooling2D(
        pool_size=3,
        strides=1,
        data_format=keras.config.image_data_format(),
    )(branch_pool)
    branch_pool = conv_block(branch_pool, 192, 1, name=f"{name}_branch_pool")

    return layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool]
    )


def inceptionv3_backbone_feature(inputs, *, data_format, return_stages=False):
    """InceptionV3 stem + 5-block backbone, returns final stage feature map.

    Args:
        inputs: Input image tensor.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_stages: If True, return a list of per-stage feature maps
            taken at natural downsample boundaries (after Pool1, after
            Conv2d_4a, after Mixed_5d, after Mixed_6e, after Mixed_7c).
            If False (default), return only the final stage map.

    Returns:
        Final stage feature tensor (after Mixed_7c), or a list of per-stage
        feature maps when ``return_stages=True``.
    """
    stages = []
    # Stem
    x = conv_block(inputs, 32, 3, strides=2, name="Conv2d_1a_3x3")
    x = conv_block(x, 32, 3, name="Conv2d_2a_3x3")
    x = conv_block(x, 64, 3, padding=None, name="Conv2d_2b_3x3")

    x = layers.MaxPooling2D(3, 2, name="Pool1")(x)
    stages.append(x)  # stage 1: after Pool1
    x = conv_block(x, 80, 1, name="Conv2d_3b_1x1")
    x = conv_block(x, 192, 3, name="Conv2d_4a_3x3")

    x = layers.MaxPooling2D(3, 2, name="Pool2")(x)
    x = inception_blocka(x, 32, "Mixed_5b")
    x = inception_blocka(x, 64, "Mixed_5c")
    x = inception_blocka(x, 64, "Mixed_5d")
    stages.append(x)  # stage 2: after Inception-A group

    x = inception_blockb(x, "Mixed_6a")
    x = inception_blockc(x, 128, "Mixed_6b")
    x = inception_blockc(x, 160, "Mixed_6c")
    x = inception_blockc(x, 160, "Mixed_6d")
    x = inception_blockc(x, 192, "Mixed_6e")
    stages.append(x)  # stage 3: after Inception-C group

    x = inception_blockd(x, "Mixed_7a")
    x = inception_blocke(x, "Mixed_7b")
    x = inception_blocke(x, "Mixed_7c")
    stages.append(x)  # stage 4: after Inception-E group

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class InceptionV3Model(BaseModel):
    """InceptionV3 trunk returning the final stage feature map.

    Output shape: ``(B, H, W, C)`` — the last stage's 4D feature map,
    unpooled and head-free. :class:`InceptionV3Classify` composes this
    model and applies GAP + Dense head to produce logits.
    """

    KMODELS_CONFIG = INCEPTIONV3_MODEL_CONFIG
    KMODELS_WEIGHTS = INCEPTIONV3_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = InceptionV3Classify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_inceptionv3_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        as_backbone=False,
        name="InceptionV3Model",
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
        x = inceptionv3_backbone_feature(
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
class InceptionV3Classify(BaseModel):
    """InceptionV3 classifier (timm-ported).

    Wraps an :class:`InceptionV3Model` backbone and applies a GAP + Dense
    head to produce class logits.

    Reference:
    - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) (CVPR 2016)

    Construction:

    >>> InceptionV3Classify.from_weights("inception_v3_tf_in1k")
    >>> InceptionV3Classify.from_weights("timm:timm/inception_v3.tf_in1k")
    """

    KMODELS_CONFIG = INCEPTIONV3_MODEL_CONFIG
    KMODELS_WEIGHTS = INCEPTIONV3_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_inceptionv3_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="InceptionV3Classify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = InceptionV3Model(
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
            name="classifier",
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
