import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer, LayerScale
from kmodels.models.resmlp.resmlp_layers import Affine
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import RESMLP_CONFIG, RESMLP_WEIGHTS
from .convert_resmlp_torch_to_keras import transfer_resmlp_weights


def resmlp_block(
    x,
    dim,
    seq_len,
    mlp_ratio=4,
    init_values=1e-4,
    drop_rate=0.0,
    block_idx=None,
):
    """A building block for the ResMLP architecture.

    Args:
        x: input tensor.
        dim: int, dimension of the input features.
        seq_len: int, length of the input sequence for cross-patch mixing.
        mlp_ratio: float, ratio of the hidden dimension in the MLP to the input
            dimension (default: 4).
        init_values: float, initial value for layer scale parameters
            (default: 1e-4).
        drop_rate: float, dropout rate to apply after dense layers (default: 0.0).
        block_idx: int or None, index of the block for naming layers (default: None).

    Returns:
        Output tensor for the block.
    """
    inputs = x

    x = Affine(name=f"blocks_{block_idx}_affine_1")(inputs)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_1")(x)
    x_t = layers.Dense(
        seq_len,
        name=f"blocks_{block_idx}_dense_1",
        kernel_initializer="glorot_uniform",
    )(x_t)
    x_t = layers.Permute((2, 1), name=f"blocks_{block_idx}_permute_2")(x_t)
    if drop_rate > 0:
        x_t = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_1")(x_t)
    x_t = LayerScale(init_values, name=f"blocks_{block_idx}_scale_1")(x_t)
    x = layers.Add(name=f"blocks_{block_idx}_add_1")([inputs, x_t])

    inputs = x
    x = Affine(name=f"blocks_{block_idx}_affine_2")(x)
    x = layers.Dense(
        dim * mlp_ratio,
        activation="gelu",
        name=f"blocks_{block_idx}_dense_2",
    )(x)
    x = layers.Dense(
        dim,
        name=f"blocks_{block_idx}_dense_3",
    )(x)
    if drop_rate > 0:
        x = layers.Dropout(drop_rate, name=f"blocks_{block_idx}_dropout_2")(x)
    x = LayerScale(init_values, name=f"blocks_{block_idx}_scale_2")(x)
    x = layers.Add(name=f"blocks_{block_idx}_add_2")([inputs, x])

    return x


def resmlp_backbone_feature(
    inputs,
    *,
    patch_size,
    embed_dim,
    depth,
    mlp_ratio,
    init_values,
    drop_path_rate,
    input_shape,
    data_format,
    return_stages=False,
):
    """ResMLP stem (patch embed) + ``depth`` ResMLP blocks + final Affine.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` or ``(B, C, H, W)``.
        patch_size: Side length of each square patch.
        embed_dim: Token (channel) embedding dimension.
        depth: Number of ResMLP blocks.
        mlp_ratio: Hidden-dim multiplier inside each block's channel MLP.
        init_values: Initial LayerScale value applied at the end of each residual branch.
        drop_path_rate: Maximum stochastic-depth-style dropout rate (scaled linearly
            with block index).
        input_shape: Image input shape used to derive sequence length.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_stages: If True, return a list of per-block (post-residual)
            outputs (one per ResMLP block, ``depth`` total). ResMLP is
            isotropic — shape is constant across blocks. If False (default),
            return the single post-final-Affine sequence.

    Returns:
        Post-Affine patch sequence of shape ``(B, num_patches, embed_dim)``,
        or a list of ``depth`` per-block outputs when ``return_stages=True``.
    """
    x = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        data_format=data_format,
        name="stem_conv",
    )(inputs)

    if data_format == "channels_first":
        if len(input_shape) == 3:
            _, height, width = input_shape
        else:
            height, width = input_shape[1:]
    else:
        if len(input_shape) == 3:
            height, width, _ = input_shape
        else:
            height, width = input_shape[:2]

    num_patches = (height // patch_size) * (width // patch_size)

    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1))(x)
    x = layers.Reshape((num_patches, embed_dim))(x)

    stages = []
    for i in range(depth):
        drop_path = drop_path_rate * (i / depth)
        x = resmlp_block(
            x,
            embed_dim,
            num_patches,
            mlp_ratio,
            init_values,
            drop_path,
            block_idx=i,
        )
        stages.append(x)

    if return_stages:
        return stages

    x = Affine(name="Final_affine")(x)
    return x


@keras.saving.register_keras_serializable(package="kmodels")
class ResMLPModel(BaseModel):
    """ResMLP backbone — the main feature extractor.

    Returns the final Affine-normalized patch sequence ``(B, N, D)``
    where ``N = (H/patch_size) * (W/patch_size)``. This is the last
    layer output before the classifier head. :class:`ResMLPClassify`
    composes this model and applies GAP1D + Dense.

    Reference:
        Touvron et al., *ResMLP: Feedforward networks for image
        classification with data-efficient training*
        (https://arxiv.org/abs/2105.03404).

    Construction:

    >>> ResMLPModel.from_weights("resmlp_12_224_fb_in1k")
    >>> ResMLPModel.from_weights("timm:timm/resmlp_12_224.fb_in1k")
    """

    KMODELS_CONFIG = RESMLP_CONFIG
    KMODELS_WEIGHTS = RESMLP_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = ResMLPClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_resmlp_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=384,
        depth=12,
        mlp_ratio=4,
        init_values=1e-4,
        drop_rate=0.0,
        drop_path_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        as_backbone=False,
        name="ResMLPModel",
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
        x = resmlp_backbone_feature(
            x,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
            drop_path_rate=drop_path_rate,
            input_shape=input_shape,
            data_format=data_format,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.init_values = init_values
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
                "depth": self.depth,
                "mlp_ratio": self.mlp_ratio,
                "init_values": self.init_values,
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
class ResMLPClassify(BaseModel):
    """ResMLP image classifier — :class:`ResMLPModel` + GAP1D + Dense.

    Wraps a :class:`ResMLPModel` backbone and attaches the standard timm
    ResMLP classifier head: global average pooling over patch tokens,
    then a single Dense layer producing class logits.

    Reference:
        Touvron et al., *ResMLP: Feedforward networks for image
        classification with data-efficient training*
        (https://arxiv.org/abs/2105.03404).

    Construction:

    >>> ResMLPClassify.from_weights("resmlp_12_224_fb_in1k")
    >>> ResMLPClassify.from_weights("timm:timm/resmlp_12_224.fb_in1k")
    """

    KMODELS_CONFIG = RESMLP_CONFIG
    KMODELS_WEIGHTS = RESMLP_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_resmlp_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        embed_dim=384,
        depth=12,
        mlp_ratio=4,
        init_values=1e-4,
        drop_rate=0.0,
        drop_path_rate=0.0,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ResMLPClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        backbone = ResMLPModel(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling1D(name="avg_pool")(backbone.output)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.init_values = init_values
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
                "depth": self.depth,
                "mlp_ratio": self.mlp_ratio,
                "init_values": self.init_values,
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
