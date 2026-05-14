import keras
import numpy as np
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer, StochasticDepth
from kmodels.models.mit.mit_layers import EfficientMultiheadSelfAttention
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import MIT_CONFIG, MIT_WEIGHTS
from .convert_mit_torch_to_keras import transfer_mit_weights


def mlp_block(x, H, W, channels, mid_channels, data_format, name_prefix):
    """Dense -> spatial DWConv -> GELU -> Dense (the MiT Mix-FFN)."""
    x = layers.Dense(mid_channels, name=f"{name_prefix}_dense_1")(x)

    input_shape = ops.shape(x)
    if data_format == "channels_first":
        x = layers.Reshape((input_shape[-1], H, W))(x)
    else:
        x = layers.Reshape((H, W, input_shape[-1]))(x)

    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        data_format=data_format,
        name=f"{name_prefix}_dwconv",
    )(x)

    x = layers.Reshape((H * W, input_shape[-1]))(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dense(channels, name=f"{name_prefix}_dense_2")(x)
    return x


def overlap_patch_embedding_block(
    x,
    channels_axis,
    data_format,
    out_channels=32,
    patch_size=7,
    stride=4,
    stage_idx=1,
):
    """Overlapping patch embedding: ZeroPad -> Conv2D(patch_size, stride) -> Reshape -> LN."""
    pytorch_stage_idx = stage_idx - 1

    x = keras.layers.ZeroPadding2D(padding=(patch_size // 2, patch_size // 2))(x)
    x = layers.Conv2D(
        filters=out_channels,
        kernel_size=patch_size,
        strides=stride,
        padding="valid",
        data_format=data_format,
        name=f"patch_embed_{pytorch_stage_idx}_conv_proj",
    )(x)
    shape = ops.shape(x)
    if data_format == "channels_first":
        H, W = shape[2], shape[3]
    else:
        H, W = shape[1], shape[2]
    x = layers.Reshape((-1, out_channels))(x)
    x = layers.LayerNormalization(
        axis=-1,
        epsilon=1e-5,
        name=f"patch_embed_{pytorch_stage_idx}_layernorm",
    )(x)
    return x, H, W


def hierarchical_transformer_encoder_block(
    x,
    H,
    W,
    project_dim,
    num_heads,
    stage_idx,
    block_idx,
    channels_axis,
    data_format,
    qkv_bias=False,
    sr_ratio=1,
    drop_prob=0.0,
):
    """LN -> efficient self-attn -> Add -> LN -> Mix-FFN -> Add."""
    pytorch_stage_idx = stage_idx - 1
    drop_path_layer = StochasticDepth(drop_prob)

    norm1 = layers.LayerNormalization(
        axis=-1,
        epsilon=1e-5,
        name=f"block_{pytorch_stage_idx}_{block_idx}_layernorm_1",
    )(x)

    attn_layer = EfficientMultiheadSelfAttention(
        project_dim,
        sr_ratio,
        block_prefix=f"block_{pytorch_stage_idx}_{block_idx}",
        qkv_bias=qkv_bias,
        num_heads=num_heads,
    )

    attn_out = attn_layer(norm1)
    attn_out = drop_path_layer(attn_out)
    add1 = layers.Add()([x, attn_out])

    norm2 = layers.LayerNormalization(
        axis=-1,
        epsilon=1e-5,
        name=f"block_{pytorch_stage_idx}_{block_idx}_layernorm_2",
    )(add1)

    mlp_out = mlp_block(
        norm2,
        H,
        W,
        channels=project_dim,
        mid_channels=int(project_dim * 4),
        data_format=data_format,
        name_prefix=f"block_{pytorch_stage_idx}_{block_idx}_mlp",
    )

    mlp_out = drop_path_layer(mlp_out)
    return layers.Add()([add1, mlp_out])


def mit_backbone_feature(
    inputs,
    *,
    embed_dims,
    depths,
    drop_path_rate,
    data_format,
    channels_axis,
):
    """MiT 4-stage encoder. Returns a list of four spatial feature maps."""
    num_stages = 4
    blockwise_num_heads = [1, 2, 5, 8]
    blockwise_sr_ratios = [8, 4, 2, 1]

    total_blocks = sum(depths)
    dpr = [x.item() for x in np.linspace(0.0, drop_path_rate, total_blocks)]

    x = inputs
    features = []
    cur_block = 0

    for i in range(num_stages):
        x, H, W = overlap_patch_embedding_block(
            x,
            out_channels=embed_dims[i],
            channels_axis=channels_axis,
            data_format=data_format,
            patch_size=7 if i == 0 else 3,
            stride=4 if i == 0 else 2,
            stage_idx=i + 1,
        )

        for j in range(depths[i]):
            x = hierarchical_transformer_encoder_block(
                x,
                H,
                W,
                project_dim=embed_dims[i],
                num_heads=blockwise_num_heads[i],
                stage_idx=i + 1,
                block_idx=j,
                sr_ratio=blockwise_sr_ratios[i],
                drop_prob=dpr[cur_block],
                qkv_bias=True,
                channels_axis=channels_axis,
                data_format=data_format,
            )
            cur_block += 1

        x = layers.LayerNormalization(
            name=f"final_layernorm_{i}", axis=-1, epsilon=1e-5
        )(x)
        if data_format == "channels_first":
            x = layers.Reshape((embed_dims[i], H, W))(x)
        else:
            x = layers.Reshape((H, W, embed_dims[i]))(x)
        features.append(x)

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class MiTClassify(BaseModel):
    """Mix Transformer (SegFormer encoder) classifier.

    Reference:
    - [SegFormer](https://arxiv.org/abs/2105.15203)

    Construction:

    >>> MiTClassify.from_weights("mit_b0_in1k")              # kmodels release
    >>> MiTClassify.from_weights("hf:nvidia/mit-b0")         # direct from HF
    """

    KMODELS_CONFIG = MIT_CONFIG
    KMODELS_WEIGHTS = MIT_WEIGHTS
    HF_MODEL_TYPE = "segformer"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "embed_dims": hf_config["hidden_sizes"],
            "depths": hf_config["depths"],
            "num_classes": hf_config.get("num_labels", 1000),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        transfer_mit_weights(keras_model, state_dict)

    def __init__(
        self,
        embed_dims=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        drop_path_rate=0.1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        num_classes=1000,
        classifier_activation="linear",
        name="MiTClassify",
        **kwargs,
    ):
        kwargs.pop("hf_id", None)

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
        features = mit_backbone_feature(
            x,
            embed_dims=embed_dims,
            depths=depths,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.embed_dims = list(embed_dims)
        self.depths = list(depths)
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
                "embed_dims": self.embed_dims,
                "depths": self.depths,
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
class MiTBackbone(BaseModel):
    """MiT feature extractor (used as the SegFormer encoder). Returns 4 stage feature maps."""

    KMODELS_CONFIG = MIT_CONFIG
    KMODELS_WEIGHTS = MIT_WEIGHTS
    HF_MODEL_TYPE = "segformer"

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = MiTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "embed_dims": hf_config["hidden_sizes"],
            "depths": hf_config["depths"],
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        transfer_mit_weights(keras_model, state_dict)

    def __init__(
        self,
        embed_dims=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        drop_path_rate=0.1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        name="MiTBackbone",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "hf_id"):
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
        features = mit_backbone_feature(
            x,
            embed_dims=embed_dims,
            depths=depths,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.embed_dims = list(embed_dims)
        self.depths = list(depths)
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dims": self.embed_dims,
                "depths": self.depths,
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
class MiTModel(BaseModel):
    """MiT trunk returning the final stage spatial feature map ``(B, H, W, C)``."""

    KMODELS_CONFIG = MIT_CONFIG
    KMODELS_WEIGHTS = MIT_WEIGHTS
    HF_MODEL_TYPE = "segformer"

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = MiTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "embed_dims": hf_config["hidden_sizes"],
            "depths": hf_config["depths"],
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        transfer_mit_weights(keras_model, state_dict)

    def __init__(
        self,
        embed_dims=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        drop_path_rate=0.1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        input_shape=None,
        name="MiTModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "hf_id"):
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
        features = mit_backbone_feature(
            x,
            embed_dims=embed_dims,
            depths=depths,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features[-1], name=name, **kwargs)

        self.embed_dims = list(embed_dims)
        self.depths = list(depths)
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dims": self.embed_dims,
                "depths": self.depths,
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
