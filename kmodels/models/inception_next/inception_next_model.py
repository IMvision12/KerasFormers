import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer, LayerScale
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import INCEPTION_NEXT_MODEL_CONFIG, INCEPTION_NEXT_WEIGHT_CONFIG
from .convert_inception_next_torch_to_keras import transfer_inception_next_weights


def inception_dwconv2d(
    x,
    square_kernel_size=3,
    band_kernel_size=11,
    branch_ratio=0.125,
    data_format=None,
    channels_axis=None,
    name="token_mixer",
):
    """Inception-style token mixer: square + band depthwise convs over channel splits.

    The input channels are split into 4 groups: an identity branch (the bulk of
    the channels), a square ``k x k`` depthwise branch, and two band ``1 x k`` /
    ``k x 1`` depthwise branches. Outputs are concatenated back along the channel
    axis.

    Args:
        x: Input feature map.
        square_kernel_size: Kernel size of the square depthwise branch.
            Defaults to ``3``.
        band_kernel_size: Length of the band depthwise branches. Defaults to ``11``.
        branch_ratio: Fraction of input channels allocated to each non-identity
            branch. Defaults to ``0.125``.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with the same shape as ``x``.
    """
    input_channels = x.shape[channels_axis]
    branch_channels = int(input_channels * branch_ratio)
    split_sizes = [input_channels - 3 * branch_channels] + [branch_channels] * 3
    split_indices = [sum(split_sizes[: i + 1]) for i in range(len(split_sizes) - 1)]

    def calculate_padding(kernel_size):
        return (kernel_size - 1) // 2

    square_padding, band_padding = (
        calculate_padding(square_kernel_size),
        calculate_padding(band_kernel_size),
    )

    x_splits = keras.ops.split(x, split_indices, axis=channels_axis)
    x_id, *x_branches = x_splits

    conv_configs = [
        (square_kernel_size, square_padding, f"{name}_dwconv_hw"),
        ((1, band_kernel_size), (0, band_padding), f"{name}_dwconv_w"),
        ((band_kernel_size, 1), (band_padding, 0), f"{name}_dwconv_h"),
    ]

    x = [
        layers.DepthwiseConv2D(
            kernel, use_bias=True, data_format=data_format, name=lname
        )(layers.ZeroPadding2D(padding)(branch_input))
        for (kernel, padding, lname), branch_input in zip(conv_configs, x_branches)
    ]

    return layers.Concatenate(axis=channels_axis)([x_id, *x])


def inception_next_block(
    x,
    num_filter,
    mlp_ratio=4.0,
    dropout_rate=0.0,
    layer_scale_init_value=1e-6,
    band_kernel_size=11,
    branch_ratio=0.125,
    data_format=None,
    channels_axis=None,
    name="blocks",
):
    """InceptionNeXt block: token mixer -> BN -> Conv MLP -> LayerScale -> residual.

    Args:
        x: Input feature map.
        num_filter: Channel count of the block (input == output).
        mlp_ratio: Hidden-dim expansion ratio for the MLP. Defaults to ``4.0``.
        dropout_rate: Dropout applied inside the MLP. Defaults to ``0.0``.
        layer_scale_init_value: Initial value for the LayerScale gamma.
            Defaults to ``1e-6``.
        band_kernel_size: Band length for the token mixer. Defaults to ``11``.
        branch_ratio: Channel fraction per token-mixer branch. Defaults to ``0.125``.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with the same shape as ``x``.
    """
    x_input = x

    x = inception_dwconv2d(
        x,
        band_kernel_size=band_kernel_size,
        branch_ratio=branch_ratio,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_token_mixer",
    )

    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, name=f"{name}_batchnorm"
    )(x)

    x = layers.Conv2D(
        int(num_filter * mlp_ratio),
        1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv1",
    )(x)
    x = layers.Activation("gelu", name=f"{name}_act")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(
        num_filter, 1, use_bias=True, data_format=data_format, name=f"{name}_conv2"
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    x = LayerScale(layer_scale_init_value, name=f"{name}_gamma")(x)
    x = layers.Add()([x, x_input])

    return x


def inception_next_backbone_feature(
    inputs,
    *,
    depths,
    num_filters,
    mlp_ratios,
    band_kernel_size,
    branch_ratio,
    data_format,
    channels_axis,
    return_stages=False,
):
    """InceptionNeXt stem + 4 stages.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        depths: Number of blocks per stage (length 4).
        num_filters: Output channel count per stage (length 4).
        mlp_ratios: MLP expansion ratio per stage (length 4).
        band_kernel_size: Band length for the token mixer (shared by all stages).
        branch_ratio: Channel fraction per token-mixer branch.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index.
        return_stages: If ``True``, return a list of the 4 per-stage feature
            maps instead of just the final one. Defaults to ``False``.

    Returns:
        Final stage feature map with ``num_filters[-1]`` channels at spatial
        resolution ``H/32`` when ``return_stages=False``. When
        ``return_stages=True``, a list of 4 per-stage feature maps.
    """
    x = layers.Conv2D(
        num_filters[0],
        4,
        4,
        use_bias=True,
        data_format=data_format,
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name="stem_batchnorm"
    )(x)

    stages = []
    for i in range(len(depths)):
        strides = 2 if i > 0 else 1
        if strides > 1:
            x = layers.BatchNormalization(
                axis=channels_axis,
                momentum=0.9,
                epsilon=1e-5,
                name=f"stages_{i}_downsample_batchnorm",
            )(x)
            x = layers.Conv2D(
                num_filters[i],
                2,
                strides,
                use_bias=True,
                data_format=data_format,
                name=f"stages_{i}_downsample_conv",
            )(x)

        for j in range(depths[i]):
            x = inception_next_block(
                x,
                num_filter=num_filters[i],
                mlp_ratio=mlp_ratios[i],
                band_kernel_size=band_kernel_size,
                branch_ratio=branch_ratio,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"stages_{i}_blocks_{j}",
            )

        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class InceptionNextModel(BaseModel):
    """InceptionNeXt backbone — the main feature extractor.

    Returns the final stage feature map ``(B, H, W, C)`` (channels-last) /
    ``(B, C, H, W)`` (channels-first), unpooled and head-free. This is the
    last layer output before the classifier head. :class:`InceptionNextClassify`
    composes this model and appends GAP + MLP head.

    Reference:
    - [InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/abs/2303.16900)

    Construction:

    >>> InceptionNextModel.from_weights("inception_next_tiny_sail_in1k")
    >>> InceptionNextModel.from_weights("timm:timm/inception_next_tiny.sail_in1k")
    """

    BASE_MODEL_CONFIG = {
        variant: INCEPTION_NEXT_MODEL_CONFIG[meta["model"]]
        for variant, meta in INCEPTION_NEXT_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = INCEPTION_NEXT_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = InceptionNextClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_inception_next_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        num_filters=(96, 192, 384, 768),
        mlp_ratios=(4, 4, 4, 3),
        band_kernel_size=11,
        branch_ratio=0.125,
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        as_backbone=False,
        name="InceptionNextModel",
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
        x = inception_next_backbone_feature(
            x,
            depths=depths,
            num_filters=num_filters,
            mlp_ratios=mlp_ratios,
            band_kernel_size=band_kernel_size,
            branch_ratio=branch_ratio,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depths = list(depths)
        self.num_filters = list(num_filters)
        self.mlp_ratios = list(mlp_ratios)
        self.band_kernel_size = band_kernel_size
        self.branch_ratio = branch_ratio
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "num_filters": self.num_filters,
                "mlp_ratios": self.mlp_ratios,
                "band_kernel_size": self.band_kernel_size,
                "branch_ratio": self.branch_ratio,
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
class InceptionNextClassify(BaseModel):
    """InceptionNeXt classifier (timm-ported).

    Wraps an :class:`InceptionNextModel` backbone and applies the MLP head:
    GAP -> Dense(num_filters[-1] * 3) -> GELU -> LayerNorm -> Dense.

    Reference:
    - [InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/abs/2303.16900)

    Construction:

    >>> InceptionNextClassify.from_weights("inception_next_tiny_sail_in1k")
    >>> InceptionNextClassify.from_weights("timm:timm/inception_next_tiny.sail_in1k")
    """

    BASE_MODEL_CONFIG = {
        variant: INCEPTION_NEXT_MODEL_CONFIG[meta["model"]]
        for variant, meta in INCEPTION_NEXT_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = INCEPTION_NEXT_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_inception_next_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        num_filters=(96, 192, 384, 768),
        mlp_ratios=(4, 4, 4, 3),
        band_kernel_size=11,
        branch_ratio=0.125,
        image_size=224,
        include_normalization=True,
        normalization_mode="inception",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="InceptionNextClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = InceptionNextModel(
            depths=depths,
            num_filters=num_filters,
            mlp_ratios=mlp_ratios,
            band_kernel_size=band_kernel_size,
            branch_ratio=branch_ratio,
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
        x = layers.Dense(int(num_filters[-1] * 3.0), use_bias=True, name="head_fc")(x)
        x = layers.Activation("gelu")(x)
        x = layers.LayerNormalization(epsilon=1e-6, name="head_batchnorm")(x)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.depths = list(depths)
        self.num_filters = list(num_filters)
        self.mlp_ratios = list(mlp_ratios)
        self.band_kernel_size = band_kernel_size
        self.branch_ratio = branch_ratio
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
                "depths": self.depths,
                "num_filters": self.num_filters,
                "mlp_ratios": self.mlp_ratios,
                "band_kernel_size": self.band_kernel_size,
                "branch_ratio": self.branch_ratio,
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
