import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import MLP_MIXER_CONFIG, MLP_MIXER_WEIGHTS
from .convert_mlpmixer_torch_to_keras import transfer_mlp_mixer_weights


def mixer_block(
    x,
    patches,
    filters,
    token_mlp_dim,
    channel_mlp_dim,
    channels_axis,
    drop_rate=0.0,
    block_idx=None,
):
    """A building block for the MLP-Mixer architecture.

    Args:
        x: input tensor.
        patches: int, the number of patches (sequence length) for token mixing.
        filters: int, the number of output filters for channel mixing.
        token_mlp_dim: int, hidden dimension for token mixing MLP.
        channel_mlp_dim: int, hidden dimension for channel mixing MLP.
        channels_axis: int, axis along which the channels are defined (-1 for
            'channels_last', 1 for 'channels_first').
        drop_rate: float, dropout rate to apply after dense layers (default: 0.0).
        block_idx: int or None, index of the block for naming layers (default: None).

    Returns:
        Output tensor for the block.
    """

    inputs = x

    x = layers.LayerNormalization(
        axis=-1, epsilon=1e-6, name=f"blocks_{block_idx}_layernorm_1"
    )(x)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_1")(x)
    x_t = layers.Dense(
        token_mlp_dim,
        name=f"blocks_{block_idx}_dense_1",
        kernel_initializer="glorot_uniform",
    )(x_t)
    x_t = layers.Activation("gelu", name=f"blocks_{block_idx}_gelu_1")(x_t)
    if drop_rate > 0:
        x_t = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_1")(x_t)
    x_t = layers.Dense(
        patches, name=f"blocks_{block_idx}_dense_2", kernel_initializer="glorot_uniform"
    )(x_t)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_2")(x_t)
    x = layers.Add(name=f"blocks_{block_idx}_add_1")([inputs, x_t])

    inputs = x
    x = layers.LayerNormalization(
        axis=-1, epsilon=1e-6, name=f"blocks_{block_idx}_layernorm_2"
    )(x)
    x = layers.Dense(
        channel_mlp_dim,
        name=f"blocks_{block_idx}_dense_3",
        kernel_initializer="glorot_uniform",
    )(x)
    x = layers.Activation("gelu", name=f"blocks_{block_idx}_gelu_2")(x)
    if drop_rate > 0:
        x = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_2")(x)
    x = layers.Dense(
        filters, name=f"blocks_{block_idx}_dense_4", kernel_initializer="glorot_uniform"
    )(x)
    x = layers.Add(name=f"blocks_{block_idx}_add_2")([inputs, x])

    return x


def _mlp_mixer_features(
    inputs,
    *,
    patch_size,
    embed_dim,
    num_blocks,
    mlp_ratio,
    drop_path_rate,
    input_shape,
    data_format,
    channels_axis,
):
    """Patch-embed + N mixer blocks + final LayerNorm, returning ``[embed, b1..bN-stride]``."""
    features = []

    x = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        data_format=data_format,
        name="stem_conv",
    )(inputs)

    if data_format == "channels_first":
        height, width = input_shape[1], input_shape[2]
        x = layers.Permute((2, 3, 1))(x)
    else:
        height, width = input_shape[0], input_shape[1]

    num_patches = (height // patch_size) * (width // patch_size)
    x = layers.Reshape((num_patches, embed_dim))(x)
    features.append(x)

    token_mlp_dim = int(embed_dim * mlp_ratio[0])
    channel_mlp_dim = int(embed_dim * mlp_ratio[1])

    features_at = [
        num_blocks // 4,
        num_blocks // 2,
        3 * num_blocks // 4,
        num_blocks - 1,
    ]
    for i in range(num_blocks):
        drop_path = drop_path_rate * (i / num_blocks)
        x = mixer_block(
            x,
            num_patches,
            embed_dim,
            token_mlp_dim,
            channel_mlp_dim,
            channels_axis,
            drop_rate=drop_path,
            block_idx=i,
        )
        if i in features_at:
            features.append(x)

    x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name="final_layernomr")(x)
    features[-1] = x

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class MLPMixerClassify(BaseModel):
    """MLP-Mixer classifier (timm-ported).

    Reference:
    - [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) (NIPS 2021)

    Construction:

    >>> MLPMixerClassify.from_weights("mixer_b16_224_goog_in21k_ft_in1k")
    >>> MLPMixerClassify.from_weights("timm:timm/mixer_b16_224.goog_in21k_ft_in1k")
    """

    KMODELS_CONFIG = MLP_MIXER_CONFIG
    KMODELS_WEIGHTS = MLP_MIXER_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mlp_mixer_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        num_blocks=12,
        mlp_ratio=(0.5, 4.0),
        drop_rate=0.0,
        drop_path_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="MLPMixerClassify",
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
        features = _mlp_mixer_features(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            input_shape=input_shape,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        x = layers.GlobalAveragePooling1D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
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
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "num_blocks": self.num_blocks,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
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
class MLPMixerBackbone(BaseModel):
    """MLP-Mixer feature extractor. Returns ``[embed, b1..b4]`` (5 maps)."""

    KMODELS_CONFIG = MLP_MIXER_CONFIG
    KMODELS_WEIGHTS = MLP_MIXER_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = MLPMixerClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mlp_mixer_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        num_blocks=12,
        mlp_ratio=(0.5, 4.0),
        drop_rate=0.0,
        drop_path_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="MLPMixerBackbone",
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
        features = _mlp_mixer_features(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            input_shape=input_shape,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "num_blocks": self.num_blocks,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
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
class MLPMixerModel(BaseModel):
    """MLP-Mixer trunk returning the final token grid as ``(B, H, W, C)``."""

    KMODELS_CONFIG = MLP_MIXER_CONFIG
    KMODELS_WEIGHTS = MLP_MIXER_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = MLPMixerClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mlp_mixer_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        num_blocks=12,
        mlp_ratio=(0.5, 4.0),
        drop_rate=0.0,
        drop_path_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="MLPMixerModel",
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
        features = _mlp_mixer_features(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            input_shape=input_shape,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        if data_format == "channels_first":
            height, width = input_shape[1], input_shape[2]
        else:
            height, width = input_shape[0], input_shape[1]
        grid_h = height // patch_size
        grid_w = width // patch_size
        out = layers.Reshape((grid_h, grid_w, embed_dim), name="final_unflatten")(
            features[-1]
        )
        if data_format == "channels_first":
            out = layers.Permute((3, 1, 2), name="final_to_cf")(out)

        super().__init__(inputs=img_input, outputs=out, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "num_blocks": self.num_blocks,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
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
