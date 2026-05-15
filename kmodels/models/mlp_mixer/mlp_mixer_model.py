import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import MLP_MIXER_MODEL_CONFIG, MLP_MIXER_WEIGHT_CONFIG
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


def mlp_mixer_backbone_feature(
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
    return_stages=False,
):
    """MLP-Mixer stem (patch embed) + N mixer blocks + final LayerNorm.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` or ``(B, C, H, W)``.
        patch_size: Side length of each square patch.
        embed_dim: Per-patch embedding (channel) dimension.
        num_blocks: Number of mixer blocks.
        mlp_ratio: Pair ``(token_mlp, channel_mlp)`` of ratios scaling
            ``embed_dim`` to the two hidden dims.
        drop_path_rate: Maximum stochastic-depth-style dropout rate (scaled linearly
            with block index).
        input_shape: Image input shape used to derive grid size.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Axis of the channel dimension.
        return_stages: If True, return a list of per-block (post-residual)
            features (one per mixer block, ``num_blocks`` total). Spatial shape
            is constant across blocks (Mixer is isotropic). If False (default),
            return the single post-final-LayerNorm sequence.

    Returns:
        Post-LayerNorm patch sequence of shape ``(B, num_patches, embed_dim)``,
        or a list of ``num_blocks`` per-block outputs when ``return_stages=True``.
    """
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

    token_mlp_dim = int(embed_dim * mlp_ratio[0])
    channel_mlp_dim = int(embed_dim * mlp_ratio[1])

    stages = []
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
        stages.append(x)

    if return_stages:
        return stages

    x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name="final_layernomr")(x)
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class MLPMixerModel(BaseModel):
    """MLP-Mixer backbone — the main feature extractor.

    Returns the final-LN normalized patch sequence ``(B, N, D)`` where
    ``N = (H/patch_size) * (W/patch_size)``. This is the last layer
    output before the classifier head. :class:`MLPMixerClassify`
    composes this model and applies GAP1D + Dense.

    Reference:
        Tolstikhin et al., *MLP-Mixer: An all-MLP Architecture for Vision*
        (https://arxiv.org/abs/2105.01601).

    Construction:

    >>> MLPMixerModel.from_weights("mixer_b16_224_goog_in21k_ft_in1k")
    >>> MLPMixerModel.from_weights("timm:timm/mixer_b16_224.goog_in21k_ft_in1k")
    """

    KMODELS_CONFIG = MLP_MIXER_MODEL_CONFIG
    KMODELS_WEIGHTS = MLP_MIXER_WEIGHT_CONFIG
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
        as_backbone=False,
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
        x = mlp_mixer_backbone_feature(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            input_shape=input_shape,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

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
        self.as_backbone = as_backbone

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
                "as_backbone": self.as_backbone,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class MLPMixerClassify(BaseModel):
    """MLP-Mixer image classifier — :class:`MLPMixerModel` + GAP1D + Dense.

    Wraps an :class:`MLPMixerModel` backbone and attaches the standard
    timm Mixer classifier head: global average pooling over patch tokens,
    then a single Dense layer producing class logits.

    Reference:
        Tolstikhin et al., *MLP-Mixer: An all-MLP Architecture for Vision*
        (https://arxiv.org/abs/2105.01601).

    Construction:

    >>> MLPMixerClassify.from_weights("mixer_b16_224_goog_in21k_ft_in1k")
    >>> MLPMixerClassify.from_weights("timm:timm/mixer_b16_224.goog_in21k_ft_in1k")
    """

    KMODELS_CONFIG = MLP_MIXER_MODEL_CONFIG
    KMODELS_WEIGHTS = MLP_MIXER_WEIGHT_CONFIG
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

        backbone = MLPMixerModel(
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling1D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

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
