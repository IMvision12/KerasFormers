import keras
import numpy as np
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer, LayerScale, StochasticDepth
from kmodels.models.convnext.convnext_layers import GlobalResponseNorm
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import CONVNEXT_CONFIG, CONVNEXT_WEIGHTS
from .convert_convnext_torch_to_keras import transfer_convnext_weights


def _spatial_layer_norm(x, data_format, epsilon=1e-6, name=None):
    """LayerNorm over channels for spatial feature maps."""
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
    """ConvNeXt block: DWConv -> LN -> (Conv|Dense) -> GELU -> (GRN) -> (Conv|Dense)."""
    x = layers.DepthwiseConv2D(
        kernel_size=7,
        padding="same",
        use_bias=True,
        data_format=data_format,
        name=name + "_depthwise_conv",
    )(inputs)
    x = _spatial_layer_norm(x, data_format, epsilon=1e-6, name=name + "_layernorm")
    if use_conv:
        x = layers.Conv2D(
            projection_dim * 4, 1, data_format=data_format, name=name + "_conv_1"
        )(x)
    else:
        x = layers.Dense(4 * projection_dim, name=name + "_dense_1")(x)
    x = layers.Activation("gelu", name=name + "_gelu")(x)
    if use_grn:
        x = GlobalResponseNorm(name=name + "_grn")(x)
    if use_conv:
        x = layers.Conv2D(
            projection_dim, 1, data_format=data_format, name=name + "_conv_2"
        )(x)
    else:
        x = layers.Dense(projection_dim, name=name + "_dense_2")(x)

    if layer_scale_init_value is not None:
        x = LayerScale(layer_scale_init_value, name=name + "_layer_scale")(x)

    if drop_path_rate:
        x = StochasticDepth(drop_path_rate, name=name + "_stochastic_depth")(x)

    return layers.Add(name=name + "_add")([inputs, x])


def _convnext_features(
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
):
    """ConvNeXt stem + 4 stages, returns ``[stem, s1, s2, s3, s4]``."""
    features = []
    x = layers.Conv2D(
        projection_dims[0],
        kernel_size=4,
        strides=4,
        data_format=data_format,
        name="stem_conv",
    )(inputs)
    x = _spatial_layer_norm(x, data_format, epsilon=1e-6, name="stem_layernorm")
    features.append(x)

    depth_drop_rates = np.linspace(0.0, drop_path_rate, sum(depths))
    cur = 0
    for i in range(len(depths)):
        if i > 0:
            x = _spatial_layer_norm(
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
        features.append(x)
        cur += depths[i]
    return features


@keras.saving.register_keras_serializable(package="kmodels")
class ConvNeXtClassify(BaseModel):
    """ConvNeXt classifier (timm-ported).

    Reference:
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

    Construction:

    >>> ConvNeXtClassify.from_weights("convnext_base_fb_in22k_ft_in1k")
    >>> ConvNeXtClassify.from_weights("timm:timm/convnext_base.fb_in22k_ft_in1k")
    """

    KMODELS_CONFIG = CONVNEXT_CONFIG
    KMODELS_WEIGHTS = CONVNEXT_WEIGHTS
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
        name="ConvNeXtClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

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
        features = _convnext_features(
            x,
            depths=depths,
            projection_dims=projection_dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            use_conv=use_conv,
            use_grn=use_grn,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name="final_layernorm")(x)
        x = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

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


@keras.saving.register_keras_serializable(package="kmodels")
class ConvNeXtBackbone(BaseModel):
    """ConvNeXt feature extractor. Returns ``[stem, s1, s2, s3, s4]``."""

    KMODELS_CONFIG = CONVNEXT_CONFIG
    KMODELS_WEIGHTS = CONVNEXT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = ConvNeXtClassify.from_weights(variant)
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
        name="ConvNeXtBackbone",
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
        features = _convnext_features(
            x,
            depths=depths,
            projection_dims=projection_dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            use_conv=use_conv,
            use_grn=use_grn,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

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
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class ConvNeXtModel(BaseModel):
    """ConvNeXt trunk returning the final stage feature map ``(B, H, W, C)``."""

    KMODELS_CONFIG = CONVNEXT_CONFIG
    KMODELS_WEIGHTS = CONVNEXT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = ConvNeXtClassify.from_weights(variant)
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
        features = _convnext_features(
            x,
            depths=depths,
            projection_dims=projection_dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            use_conv=use_conv,
            use_grn=use_grn,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features[-1], name=name, **kwargs)

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
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
