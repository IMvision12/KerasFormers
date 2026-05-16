import keras
import numpy as np
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kerasformers.base import BaseModel
from kerasformers.layers import ImageNormalizationLayer, LayerScale, StochasticDepth
from kerasformers.models.convnext.convnext_layers import GlobalResponseNorm
from kerasformers.weight_utils import copy_weights_by_path_suffix

from .config import CONVNEXT_MODEL_CONFIG, CONVNEXT_WEIGHT_CONFIG
from .convert_convnext_torch_to_keras import transfer_convnext_weights


def spatial_layer_norm(x, data_format, epsilon=1e-6, name=None):
    """LayerNorm over channels for spatial feature maps.

    Args:
        x: Input feature tensor with spatial dims.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        epsilon: Small constant added to the variance for numerical stability.
        name: Optional name prefix for the underlying layers.

    Returns:
        Tensor with the same shape as ``x`` after channel-wise LayerNorm.
    """
    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1), name=f"{name}_to_cl" if name else None)(x)
    x = layers.LayerNormalization(axis=-1, epsilon=epsilon, name=name)(x)
    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2), name=f"{name}_to_cf" if name else None)(x)
    return x


def convnext_block(
    inputs,
    projection_dim,
    channels_axis,
    data_format,
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    name=None,
    use_grn=False,
    use_conv=False,
):
    """ConvNeXt block: DWConv -> LN -> (Conv|Dense) -> GELU -> (GRN) -> (Conv|Dense).

    Args:
        inputs: Input feature map for the residual block.
        projection_dim: Output channel count of the block.
        channels_axis: Axis index of the channels dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        drop_path_rate: Stochastic depth drop probability for this block.
        layer_scale_init_value: Initial value for LayerScale; pass ``None`` to skip.
        name: Name prefix for sub-layers inside the block.
        use_grn: Whether to apply GlobalResponseNorm (ConvNeXtV2 style).
        use_conv: If True, use 1x1 Conv2D for the MLP; else use Dense layers.

    Returns:
        Output tensor with shape matching ``inputs`` and ``projection_dim`` channels.
    """
    x = layers.DepthwiseConv2D(
        kernel_size=7,
        padding="same",
        use_bias=True,
        data_format=data_format,
        name=name + "_depthwise_conv",
    )(inputs)

    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1), name=name + "_to_nhwc")(x)

    x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name=name + "_layernorm")(x)
    if use_conv:
        x = layers.Conv2D(
            projection_dim * 4,
            1,
            data_format="channels_last",
            name=name + "_conv_1",
        )(x)
    else:
        x = layers.Dense(4 * projection_dim, name=name + "_dense_1")(x)
    x = layers.Activation("gelu", name=name + "_gelu")(x)
    if use_grn:
        x = GlobalResponseNorm(name=name + "_grn")(x)
    if use_conv:
        x = layers.Conv2D(
            projection_dim,
            1,
            data_format="channels_last",
            name=name + "_conv_2",
        )(x)
    else:
        x = layers.Dense(projection_dim, name=name + "_dense_2")(x)

    if layer_scale_init_value is not None:
        x = LayerScale(layer_scale_init_value, name=name + "_layer_scale")(x)

    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2), name=name + "_to_nchw")(x)

    if drop_path_rate:
        x = StochasticDepth(drop_path_rate, name=name + "_stochastic_depth")(x)

    return layers.Add(name=name + "_add")([inputs, x])


def convnext_backbone_feature(
    inputs,
    *,
    depths,
    projection_dims,
    drop_path_rate,
    layer_scale_init_value,
    use_conv,
    use_grn,
    data_format,
    channels_axis,
    return_stages=False,
):
    """ConvNeXt stem + 4 stages.

    Args:
        inputs: Input image tensor (post-normalization).
        depths: Number of blocks per stage (length-4 list).
        projection_dims: Channel count per stage (length-4 list).
        drop_path_rate: Maximum stochastic-depth rate; linearly scaled across blocks.
        layer_scale_init_value: LayerScale init; pass ``None`` to disable.
        use_conv: Use 1x1 Conv2D inside blocks instead of Dense.
        use_grn: Enable GlobalResponseNorm (ConvNeXtV2 style).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Axis index of the channels dimension.
        return_stages: If True, return a list of per-stage feature maps
            (one per ConvNeXt stage). If False (default), return only the
            final stage feature map.

    Returns:
        Final stage feature map ``(B, H, W, C)``, or a list of per-stage
        feature maps when ``return_stages=True``.
    """
    x = layers.Conv2D(
        projection_dims[0],
        kernel_size=4,
        strides=4,
        data_format=data_format,
        name="stem_conv",
    )(inputs)
    x = spatial_layer_norm(x, data_format, epsilon=1e-6, name="stem_layernorm")

    depth_drop_rates = np.linspace(0.0, drop_path_rate, sum(depths))
    cur = 0
    stages = []
    for i in range(len(depths)):
        if i > 0:
            x = spatial_layer_norm(
                x,
                data_format,
                epsilon=1e-6,
                name=f"stages_{i}_downsampling_layernorm",
            )
            x = layers.Conv2D(
                projection_dims[i],
                kernel_size=2,
                strides=2,
                data_format=data_format,
                name=f"stages_{i}_downsampling_conv",
            )(x)
        for j in range(depths[i]):
            x = convnext_block(
                x,
                projection_dim=projection_dims[i],
                drop_path_rate=depth_drop_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value,
                use_grn=use_grn,
                use_conv=use_conv,
                channels_axis=channels_axis,
                data_format=data_format,
                name=f"stages_{i}_blocks_{j}",
            )
        cur += depths[i]
        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kerasformers")
class ConvNeXtModel(BaseModel):
    """Instantiates the ConvNeXt backbone.

    ConvNeXt is a modernized ConvNet that adapts ViT design principles to
    pure CNNs: depthwise 7x7 convolutions, LayerNorm in place of
    BatchNorm, GELU activations, and inverted-bottleneck blocks organized
    into 4 hierarchical stages. Output is the last layer output before
    the classifier head: the final stage feature map ``(B, H, W, C)``.
    :class:`ConvNeXtImageClassify` composes this model and attaches a
    GlobalAveragePooling2D + LayerNorm + Dense head to produce logits.

    References:
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps (one per ConvNeXt stage).
            Defaults to `False`.
        depths: Tuple of 4 integers, number of ConvNeXt blocks per stage.
            Defaults to `(3, 3, 9, 3)`.
        projection_dims: Tuple of 4 integers, channel count per stage.
            Defaults to `(96, 192, 384, 768)`.
        drop_path_rate: Float, maximum stochastic-depth drop rate.
            Linearly scaled from 0 to this value across all blocks.
            Defaults to `0.0`.
        layer_scale_init_value: Float, initial value for per-channel
            LayerScale. Pass ``None`` to disable LayerScale.
            Defaults to `1e-6`.
        use_conv: Boolean, if True, use 1x1 Conv2D layers inside each
            block's MLP; otherwise use Dense layers. Defaults to `False`.
        use_grn: Boolean, whether to apply GlobalResponseNorm inside each
            block (ConvNeXtV2 recipe). Defaults to `False`.
        image_size: Integer, square input resolution used to validate the
            input shape. Defaults to `224`.
        include_normalization: Boolean, whether to prepend an
            :class:`~kerasformers.layers.ImageNormalizationLayer` at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'` (default), `'inception'`,
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_shape: Optional tuple specifying the shape of the input
            data. If `None`, derived from ``image_size`` and the active
            Keras data format. Defaults to `None`.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        name: String, the name of the model. Defaults to `"ConvNeXtModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = {
        variant: CONVNEXT_MODEL_CONFIG[meta["model"]]
        for variant, meta in CONVNEXT_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = CONVNEXT_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = ConvNeXtImageClassify.from_weights(
                variant, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_convnext_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        projection_dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        use_conv=False,
        use_grn=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        as_backbone=False,
        name="ConvNeXtModel",
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
        x = convnext_backbone_feature(
            x,
            depths=depths,
            projection_dims=projection_dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            use_conv=use_conv,
            use_grn=use_grn,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depths = list(depths)
        self.projection_dims = list(projection_dims)
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.use_conv = use_conv
        self.use_grn = use_grn
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
                "projection_dims": self.projection_dims,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init_value": self.layer_scale_init_value,
                "use_conv": self.use_conv,
                "use_grn": self.use_grn,
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


@keras.saving.register_keras_serializable(package="kerasformers")
class ConvNeXtImageClassify(BaseModel):
    """Instantiates the ConvNeXt classifier.

    This classifier wraps a :class:`ConvNeXtModel` backbone and attaches
    a GlobalAveragePooling2D + LayerNorm + Dense head to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`ConvNeXtModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

    Args:
        depths: Tuple of 4 integers, number of ConvNeXt blocks per stage.
            Defaults to `(3, 3, 9, 3)`.
        projection_dims: Tuple of 4 integers, channel count per stage.
            Defaults to `(96, 192, 384, 768)`.
        drop_path_rate: Float, maximum stochastic-depth drop rate.
            Linearly scaled from 0 to this value across all blocks.
            Defaults to `0.0`.
        layer_scale_init_value: Float, initial value for per-channel
            LayerScale. Pass ``None`` to disable LayerScale.
            Defaults to `1e-6`.
        use_conv: Boolean, if True, use 1x1 Conv2D layers inside each
            block's MLP; otherwise use Dense layers. Defaults to `False`.
        use_grn: Boolean, whether to apply GlobalResponseNorm inside each
            block (ConvNeXtV2 recipe). Defaults to `False`.
        image_size: Integer, square input resolution used to validate the
            input shape. Defaults to `224`.
        include_normalization: Boolean, whether to prepend an
            :class:`~kerasformers.layers.ImageNormalizationLayer` at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'` (default), `'inception'`,
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_shape: Optional tuple specifying the shape of the input
            data. If `None`, derived from ``image_size`` and the active
            Keras data format. Defaults to `None`.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        num_classes: Integer, the number of output classes for
            classification. Defaults to `1000`.
        classifier_activation: String or callable, activation function
            for the final Dense layer. Use `"linear"` to return raw
            logits or `"softmax"` to return class probabilities.
            Defaults to `"linear"`.
        name: String, the name of the model. The internal backbone is
            named `f"{name}_backbone"`. Defaults to `"ConvNeXtImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = {
        variant: CONVNEXT_MODEL_CONFIG[meta["model"]]
        for variant, meta in CONVNEXT_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = CONVNEXT_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_convnext_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        projection_dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        use_conv=False,
        use_grn=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ConvNeXtImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = ConvNeXtModel(
            depths=depths,
            projection_dims=projection_dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            use_conv=use_conv,
            use_grn=use_grn,
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
        x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name="final_layernorm")(x)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.depths = list(depths)
        self.projection_dims = list(projection_dims)
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.use_conv = use_conv
        self.use_grn = use_grn
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
                "projection_dims": self.projection_dims,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init_value": self.layer_scale_init_value,
                "use_conv": self.use_conv,
                "use_grn": self.use_grn,
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
