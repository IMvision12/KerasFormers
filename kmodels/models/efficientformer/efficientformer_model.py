"""EfficientFormer classifier and backbone (timm-ported)."""

import keras
import numpy as np
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer, LayerScale, StochasticDepth
from kmodels.models.efficientformer.efficientformer_layers import Attention4D
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import EFFICIENTFORMER_CONFIG, EFFICIENTFORMER_WEIGHTS
from .convert_efficientformer_torch_to_keras import transfer_efficientformer_weights


def conv_mlp_block(
    inputs,
    hidden_features,
    out_features,
    drop=0.0,
    channels_axis=-1,
    data_format="channels_last",
    name=None,
):
    """MLP block with 1x1 convolutions for 2D spatial feature maps."""
    x = layers.Conv2D(
        hidden_features,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_1",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, name=f"{name}_norm_1"
    )(x)
    x = layers.Activation("gelu", name=f"{name}_gelu_1")(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_1")(x)

    x = layers.Conv2D(
        out_features,
        kernel_size=1,
        use_bias=True,
        data_format=data_format,
        name=f"{name}_conv_2",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, name=f"{name}_norm_2"
    )(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_2")(x)
    return x


def mlp_block(inputs, hidden_features, out_features, drop=0.0, name=None):
    """Standard MLP block for 1D token sequences."""
    x = layers.Dense(hidden_features, use_bias=True, name=f"{name}_dense_1")(inputs)
    x = layers.Activation("gelu", name=f"{name}_gelu")(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_1")(x)
    x = layers.Dense(out_features, use_bias=True, name=f"{name}_dense_2")(x)
    x = layers.Dropout(drop, name=f"{name}_dropout_2")(x)
    return x


def meta_block_2d(
    inputs,
    dim,
    pool_size=3,
    mlp_ratio=4.0,
    drop=0.0,
    drop_path=0.0,
    layer_scale_init_value=1e-5,
    channels_axis=-1,
    data_format="channels_last",
    name=None,
):
    """2D MetaBlock with pooling token mixer for convolutional stages."""
    # Token mixer (pooling)
    pooled = layers.AveragePooling2D(
        pool_size=pool_size,
        strides=1,
        padding="same",
        data_format=data_format,
        name=f"{name}_pool_pool",
    )(inputs)
    x = layers.Subtract(name=f"{name}_pool_sub")([pooled, inputs])
    x = LayerScale(layer_scale_init_value, name=f"{name}_ls1")(x)
    if drop_path > 0.0:
        x = StochasticDepth(drop_path, name=f"{name}_drop_path1")(x)
    x = layers.Add(name=f"{name}_add1")([inputs, x])

    # MLP
    y = conv_mlp_block(
        x,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        drop=drop,
        channels_axis=channels_axis,
        data_format=data_format,
        name=f"{name}_mlp",
    )
    y = LayerScale(layer_scale_init_value, name=f"{name}_ls2")(y)
    if drop_path > 0.0:
        y = StochasticDepth(drop_path, name=f"{name}_drop_path2")(y)
    outputs = layers.Add(name=f"{name}_add2")([x, y])
    return outputs


def meta_block_1d(
    inputs,
    dim,
    mlp_ratio=4.0,
    drop=0.0,
    drop_path=0.0,
    layer_scale_init_value=1e-5,
    resolution=7,
    name=None,
):
    """1D MetaBlock with self-attention token mixer for transformer stages."""
    y = layers.LayerNormalization(epsilon=1e-6, axis=-1, name=f"{name}_norm1")(inputs)
    y = Attention4D(dim=dim, resolution=resolution, name=f"{name}_attn")(y)
    y = LayerScale(layer_scale_init_value, name=f"{name}_ls1")(y)
    if drop_path > 0.0:
        y = StochasticDepth(drop_path, name=f"{name}_drop_path1")(y)
    x = layers.Add(name=f"{name}_add1")([inputs, y])

    # MLP
    y = layers.LayerNormalization(epsilon=1e-6, axis=-1, name=f"{name}_norm2")(x)
    y = mlp_block(
        y,
        hidden_features=int(dim * mlp_ratio),
        out_features=dim,
        drop=drop,
        name=f"{name}_mlp",
    )
    y = LayerScale(layer_scale_init_value, name=f"{name}_ls2")(y)
    if drop_path > 0.0:
        y = StochasticDepth(drop_path, name=f"{name}_drop_path2")(y)
    outputs = layers.Add(name=f"{name}_add2")([x, y])
    return outputs


def _efficientformer_features(
    inputs,
    *,
    depths,
    embed_dims,
    num_vit,
    mlp_ratio,
    pool_size,
    drop_rate,
    drop_path_rate,
    layer_scale_init_value,
    image_h,
    data_format,
    channels_axis,
):
    """EfficientFormer stem + four hybrid stages.

    Returns one feature tensor per stage (4 tensors). The last tensor is
    1D ``(B, N, C)`` if the final stage has any transformer (vit) blocks,
    otherwise 2D like the earlier stages. Shared by :class:`EfficientFormer`
    and :class:`EfficientFormerBackbone`.
    """
    features = []

    x = layers.ZeroPadding2D(padding=1, data_format=data_format, name="stem_pad1")(
        inputs
    )
    x = layers.Conv2D(
        embed_dims[0] // 2,
        kernel_size=3,
        strides=2,
        padding="valid",
        data_format=data_format,
        name="stem_conv1",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, epsilon=1e-5, name="stem_norm1")(
        x
    )
    x = layers.Activation("relu", name="stem_act1")(x)

    x = layers.ZeroPadding2D(padding=1, data_format=data_format, name="stem_pad2")(x)
    x = layers.Conv2D(
        embed_dims[0],
        kernel_size=3,
        strides=2,
        padding="valid",
        data_format=data_format,
        name="stem_conv2",
    )(x)
    x = layers.BatchNormalization(axis=channels_axis, epsilon=1e-5, name="stem_norm2")(
        x
    )
    x = layers.Activation("relu", name="stem_act2")(x)

    num_stages = len(depths)
    dpr = np.linspace(0.0, drop_path_rate, sum(depths))
    cur = 0

    for i in range(num_stages):
        if i > 0:
            x = layers.ZeroPadding2D(
                padding=1,
                data_format=data_format,
                name=f"stages_{i}_downsample_pad",
            )(x)
            x = layers.Conv2D(
                embed_dims[i],
                kernel_size=3,
                strides=2,
                padding="valid",
                data_format=data_format,
                name=f"stages_{i}_downsample_conv",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                name=f"stages_{i}_downsample_norm",
            )(x)

        is_last_stage = i == num_stages - 1
        use_transformer = is_last_stage and num_vit > 0

        for j in range(depths[i]):
            remain_idx = depths[i] - j - 1

            if use_transformer and num_vit > remain_idx and j == depths[i] - num_vit:
                resolution = image_h // (4 * (2**i))
                ch = x.shape[channels_axis]
                if data_format == "channels_first":
                    x = layers.Permute((2, 3, 1), name=f"stages_{i}_to_nhwc")(x)
                x = layers.Reshape((-1, ch), name=f"stages_{i}_flat")(x)

            if use_transformer and num_vit > remain_idx:
                x = meta_block_1d(
                    x,
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    resolution=resolution,
                    name=f"stages_{i}_blocks_{j}",
                )
            else:
                x = meta_block_2d(
                    x,
                    dim=embed_dims[i],
                    pool_size=pool_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    channels_axis=channels_axis,
                    data_format=data_format,
                    name=f"stages_{i}_blocks_{j}",
                )

        features.append(x)
        cur += depths[i]

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class EfficientFormerClassify(BaseModel):
    """EfficientFormer classifier (timm-ported).

    Reference:
    - [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

    Construction:

    >>> EfficientFormerClassify.from_weights("efficientformer_l1_snap_dist_in1k")
    >>> EfficientFormerClassify.from_weights("timm:timm/efficientformer_l1.snap_dist_in1k")
    """

    KMODELS_CONFIG = EFFICIENTFORMER_CONFIG
    KMODELS_WEIGHTS = EFFICIENTFORMER_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_efficientformer_weights(keras_model, state_dict)

    def __init__(
        self,
        depths,
        embed_dims,
        num_vit=1,
        mlp_ratio=4.0,
        pool_size=3,
        drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="EfficientFormerClassify",
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

        if data_format == "channels_last":
            image_h = input_shape[0]
        else:
            image_h = input_shape[1]

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
        features = _efficientformer_features(
            x,
            depths=depths,
            embed_dims=embed_dims,
            num_vit=num_vit,
            mlp_ratio=mlp_ratio,
            pool_size=pool_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            image_h=image_h,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        x = features[-1]
        x = layers.LayerNormalization(epsilon=1e-6, axis=-1, name="final_norm")(x)
        x = layers.Lambda(lambda v: ops.mean(v, axis=1), name="global_pool")(x)
        x = layers.Dropout(drop_rate, name="head_drop")(x)

        x_cls = layers.Dense(num_classes, activation=None, name="head", use_bias=True)(
            x
        )
        x_dist = layers.Dense(
            num_classes, activation=None, name="head_dist", use_bias=True
        )(x)

        x = layers.Average(name="avg_predictions")([x_cls, x_dist])
        if classifier_activation:
            x = layers.Activation(classifier_activation, name="predictions")(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depths = depths
        self.embed_dims = embed_dims
        self.num_vit = num_vit
        self.mlp_ratio = mlp_ratio
        self.pool_size = pool_size
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
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
                "embed_dims": self.embed_dims,
                "num_vit": self.num_vit,
                "mlp_ratio": self.mlp_ratio,
                "pool_size": self.pool_size,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init_value": self.layer_scale_init_value,
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
class EfficientFormerBackbone(BaseModel):
    """EfficientFormer feature extractor. Returns one feature tensor per stage."""

    KMODELS_CONFIG = EFFICIENTFORMER_CONFIG
    KMODELS_WEIGHTS = EFFICIENTFORMER_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        return EfficientFormerClassify

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = cls._release_warm_start_cls().from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_efficientformer_weights(keras_model, state_dict)

    def __init__(
        self,
        depths,
        embed_dims,
        num_vit=1,
        mlp_ratio=4.0,
        pool_size=3,
        drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="EfficientFormerBackbone",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        # Attention biases in the final stage are sized to ``image_size /
        # (4 * 2**(num_stages-1))``, so we cannot allow dynamic spatial
        # dims to propagate. Build the input shape from ``image_size``
        # whenever the caller didn't provide one.
        input_shape = imagenet_utils.obtain_input_shape(
            input_shape,
            default_size=image_size,
            min_size=32,
            data_format=data_format,
            require_flatten=True,
            weights=None,
        )

        if data_format == "channels_last":
            image_h = input_shape[0]
        else:
            image_h = input_shape[1]

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
        features = _efficientformer_features(
            x,
            depths=depths,
            embed_dims=embed_dims,
            num_vit=num_vit,
            mlp_ratio=mlp_ratio,
            pool_size=pool_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            image_h=image_h,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.depths = depths
        self.embed_dims = embed_dims
        self.num_vit = num_vit
        self.mlp_ratio = mlp_ratio
        self.pool_size = pool_size
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "embed_dims": self.embed_dims,
                "num_vit": self.num_vit,
                "mlp_ratio": self.mlp_ratio,
                "pool_size": self.pool_size,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init_value": self.layer_scale_init_value,
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
class EfficientFormerModel(BaseModel):
    """EfficientFormer trunk returning the final feature map ``(B, H, W, C)``.

    If the final stage contains transformer blocks, the 1D token sequence is
    reshaped back to a 4D grid before being returned.
    """

    KMODELS_CONFIG = EFFICIENTFORMER_CONFIG
    KMODELS_WEIGHTS = EFFICIENTFORMER_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        return EfficientFormerClassify

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = cls._release_warm_start_cls().from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_efficientformer_weights(keras_model, state_dict)

    def __init__(
        self,
        depths,
        embed_dims,
        num_vit=1,
        mlp_ratio=4.0,
        pool_size=3,
        drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="EfficientFormerModel",
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

        if data_format == "channels_last":
            image_h = input_shape[0]
        else:
            image_h = input_shape[1]

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
        features = _efficientformer_features(
            x,
            depths=depths,
            embed_dims=embed_dims,
            num_vit=num_vit,
            mlp_ratio=mlp_ratio,
            pool_size=pool_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            image_h=image_h,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        last = features[-1]
        # If last stage is 1D (transformer tokens), reshape back to (B, H, W, C).
        if len(last.shape) == 3:
            num_stages = len(depths)
            grid = image_h // (4 * (2 ** (num_stages - 1)))
            ch = embed_dims[-1]
            last = layers.Reshape((grid, grid, ch), name="final_unflatten")(last)
            if data_format == "channels_first":
                last = layers.Permute((3, 1, 2), name="final_to_cf")(last)

        super().__init__(inputs=img_input, outputs=last, name=name, **kwargs)

        self.depths = depths
        self.embed_dims = embed_dims
        self.num_vit = num_vit
        self.mlp_ratio = mlp_ratio
        self.pool_size = pool_size
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "embed_dims": self.embed_dims,
                "num_vit": self.num_vit,
                "mlp_ratio": self.mlp_ratio,
                "pool_size": self.pool_size,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "layer_scale_init_value": self.layer_scale_init_value,
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
