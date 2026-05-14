import keras
from keras import layers, ops, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import NEXTVIT_CONFIG, NEXTVIT_WEIGHTS
from .convert_nextvit_timm_to_keras import transfer_nextvit_weights
from .nextvit_layers import EfficientAttention


def nextvit_conv_attention(x, out_chs, head_dim, channels_axis, data_format, prefix=""):
    """Multi-Head Convolutional Attention (MHCA)."""
    num_groups = out_chs // head_dim
    out = layers.Conv2D(
        out_chs,
        3,
        strides=1,
        padding="same",
        groups=num_groups,
        use_bias=False,
        data_format=data_format,
        name=prefix + "mhca_group_conv3x3",
    )(x)
    out = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "mhca_norm",
    )(out)
    out = layers.ReLU()(out)
    out = layers.Conv2D(
        out_chs,
        1,
        use_bias=False,
        data_format=data_format,
        name=prefix + "mhca_projection",
    )(out)
    return out


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _calculate_drop_path_rates(drop_path_rate, depths):
    total_depth = sum(depths)
    rates = []
    idx = 0
    for d in depths:
        stage_rates = []
        for i in range(d):
            stage_rates.append(
                drop_path_rate * idx / (total_depth - 1) if total_depth > 1 else 0.0
            )
            idx += 1
        rates.append(stage_rates)
    return rates


def _get_stage_out_chs(depths):
    return [
        [96] * depths[0],
        [192] * (depths[1] - 1) + [256],
        [384, 384, 384, 384, 512] * (depths[2] // 5),
        [768] * (depths[3] - 1) + [1024],
    ]


def _get_stage_block_types(depths):
    return [
        ["conv"] * depths[0],
        ["conv"] * (depths[1] - 1) + ["transformer"],
        ["conv", "conv", "conv", "conv", "transformer"] * (depths[2] // 5),
        ["conv"] * (depths[3] - 1) + ["transformer"],
    ]


def conv_mlp(
    x, in_features, hidden_features, out_features, channels_axis, data_format, prefix=""
):
    """ConvMlp block with two 1x1 convolutions and ReLU activation."""
    x = layers.Conv2D(
        hidden_features,
        1,
        use_bias=True,
        data_format=data_format,
        name=prefix + "mlp_fc1",
    )(x)
    x = layers.Activation("relu", name=prefix + "mlp_act")(x)
    x = layers.Conv2D(
        out_features,
        1,
        use_bias=True,
        data_format=data_format,
        name=prefix + "mlp_fc2",
    )(x)
    return x


def patch_embed_block(
    x, in_chs, out_chs, use_pool, channels_axis, data_format, prefix=""
):
    """Patch embedding with optional average pooling and 1x1 projection."""
    if use_pool:
        x = layers.AveragePooling2D(
            pool_size=2,
            strides=2,
            padding="valid",
            data_format=data_format,
            name=prefix + "patch_embed_pool",
        )(x)
    if use_pool or in_chs != out_chs:
        x = layers.Conv2D(
            out_chs,
            1,
            use_bias=False,
            data_format=data_format,
            name=prefix + "patch_embed_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.9,
            name=prefix + "patch_embed_norm",
        )(x)
    return x


def next_conv_block(
    x,
    in_chs,
    out_chs,
    stride,
    drop_path_rate,
    head_dim,
    mlp_ratio,
    channels_axis,
    data_format,
    prefix="",
):
    """NextConvBlock with patch embedding, MHCA, and ConvMLP."""
    use_pool = stride == 2
    x = patch_embed_block(
        x, in_chs, out_chs, use_pool, channels_axis, data_format, prefix=prefix
    )
    mhca_out = nextvit_conv_attention(
        x, out_chs, head_dim, channels_axis, data_format, prefix=prefix
    )
    x = layers.Add()([x, mhca_out])

    residual = x
    out = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "norm",
    )(x)
    out = conv_mlp(
        out,
        out_chs,
        int(out_chs * mlp_ratio),
        out_chs,
        channels_axis,
        data_format,
        prefix=prefix,
    )
    x = layers.Add()([residual, out])
    return x


def next_transformer_block(
    x,
    in_chs,
    out_chs,
    stride,
    drop_path_rate,
    head_dim,
    sr_ratio,
    mix_block_ratio,
    mlp_ratio,
    channels_axis,
    data_format,
    prefix="",
):
    """NextTransformerBlock with E-MHSA and MHCA branches."""
    mhsa_out_chs = _make_divisible(int(out_chs * mix_block_ratio), 32)
    mhca_out_chs = out_chs - mhsa_out_chs

    use_pool = stride == 2
    x = patch_embed_block(
        x, in_chs, mhsa_out_chs, use_pool, channels_axis, data_format, prefix=prefix
    )

    out = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "norm1",
    )(x)
    if data_format == "channels_first":
        out = layers.Permute((2, 3, 1), name=prefix + "to_seq_perm")(out)
    out = layers.Reshape((-1, mhsa_out_chs), name=prefix + "reshape_to_seq")(out)

    out = EfficientAttention(
        mhsa_out_chs,
        head_dim=head_dim,
        sr_ratio=sr_ratio,
        prefix=prefix,
        name=prefix + "e_mhsa",
    )(out)

    x_shape = ops.shape(x)
    if data_format == "channels_first":
        h_idx, w_idx = 2, 3
    else:
        h_idx, w_idx = 1, 2
    out = layers.Reshape(
        (x_shape[h_idx], x_shape[w_idx], mhsa_out_chs),
        name=prefix + "reshape_to_spatial",
    )(out)
    if data_format == "channels_first":
        out = layers.Permute((3, 1, 2), name=prefix + "from_seq_perm")(out)

    x = layers.Add()([x, out])

    proj_out = layers.Conv2D(
        mhca_out_chs,
        1,
        use_bias=False,
        data_format=data_format,
        name=prefix + "projection_conv",
    )(x)
    proj_out = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "projection_norm",
    )(proj_out)

    mhca_out = nextvit_conv_attention(
        proj_out,
        mhca_out_chs,
        head_dim,
        channels_axis,
        data_format,
        prefix=prefix,
    )
    proj_out = layers.Add()([proj_out, mhca_out])

    x = layers.Concatenate(axis=channels_axis)([x, proj_out])

    residual = x
    out = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "norm2",
    )(x)
    out = conv_mlp(
        out,
        out_chs,
        int(out_chs * mlp_ratio),
        out_chs,
        channels_axis,
        data_format,
        prefix=prefix,
    )
    x = layers.Add()([residual, out])
    return x


def _nextvit_features(
    inputs,
    *,
    depths,
    stem_chs,
    head_dim,
    mix_block_ratio,
    sr_ratios,
    drop_path_rate,
    data_format,
    channels_axis,
):
    """Stem + 4 stages + final BN, returning ``[stem, s1..s4]`` (5 maps)."""
    features = []

    x = inputs
    stem_configs = [
        (3, stem_chs[0], 2),
        (stem_chs[0], stem_chs[1], 1),
        (stem_chs[1], stem_chs[2], 1),
        (stem_chs[2], stem_chs[2], 2),
    ]
    for i, (in_c, out_c, stride) in enumerate(stem_configs):
        if stride == 2:
            x = layers.ZeroPadding2D(
                padding=1,
                data_format=data_format,
                name=f"stem_{i}_pad",
            )(x)
        x = layers.Conv2D(
            out_c,
            3,
            strides=stride,
            padding="valid" if stride == 2 else "same",
            use_bias=False,
            data_format=data_format,
            name=f"stem_{i}_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.9,
            name=f"stem_{i}_norm",
        )(x)
        x = layers.Activation("relu", name=f"stem_{i}_act")(x)
    features.append(x)

    stage_out_chs = _get_stage_out_chs(depths)
    stage_block_types = _get_stage_block_types(depths)
    dpr = _calculate_drop_path_rates(drop_path_rate, depths)
    strides = [1, 2, 2, 2]

    in_chs = stem_chs[-1]

    for stage_idx in range(4):
        block_chs = stage_out_chs[stage_idx]
        block_types = stage_block_types[stage_idx]

        for block_idx in range(depths[stage_idx]):
            stride = strides[stage_idx] if block_idx == 0 else 1
            out_chs = block_chs[block_idx]
            block_type = block_types[block_idx]
            dp_rate = dpr[stage_idx][block_idx]
            prefix = f"stages_{stage_idx}_blocks_{block_idx}_"

            if block_type == "conv":
                x = next_conv_block(
                    x,
                    in_chs,
                    out_chs,
                    stride,
                    dp_rate,
                    head_dim,
                    3.0,
                    channels_axis,
                    data_format,
                    prefix=prefix,
                )
            else:
                x = next_transformer_block(
                    x,
                    in_chs,
                    out_chs,
                    stride,
                    dp_rate,
                    head_dim,
                    sr_ratios[stage_idx],
                    mix_block_ratio,
                    2.0,
                    channels_axis,
                    data_format,
                    prefix=prefix,
                )
            in_chs = out_chs
        features.append(x)

    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name="norm",
    )(x)
    features[-1] = x

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class NextViTClassify(BaseModel):
    """NextViT classifier (timm-ported).

    A hybrid CNN-Transformer combining MHCA blocks with E-MHSA blocks.

    Reference:
    - [Next-ViT](https://arxiv.org/abs/2207.05501)

    Construction:

    >>> NextViT.from_weights("nextvit_small_bd_in1k")
    >>> NextViT.from_weights("timm:timm/nextvit_small.bd_in1k")
    """

    KMODELS_CONFIG = NEXTVIT_CONFIG
    KMODELS_WEIGHTS = NEXTVIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_nextvit_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 4, 10, 3),
        stem_chs=(64, 32, 64),
        head_dim=32,
        mix_block_ratio=0.75,
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="NextViTClassify",
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
        features = _nextvit_features(
            x,
            depths=depths,
            stem_chs=stem_chs,
            head_dim=head_dim,
            mix_block_ratio=mix_block_ratio,
            sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        x = layers.GlobalAveragePooling2D(
            data_format=data_format,
            name="head_global_pool",
        )(features[-1])
        x = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="head_fc",
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depths = list(depths)
        self.stem_chs = list(stem_chs)
        self.head_dim = head_dim
        self.mix_block_ratio = mix_block_ratio
        self.sr_ratios = list(sr_ratios)
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
                "depths": self.depths,
                "stem_chs": self.stem_chs,
                "head_dim": self.head_dim,
                "mix_block_ratio": self.mix_block_ratio,
                "sr_ratios": self.sr_ratios,
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
class NextViTModel(BaseModel):
    """NextViT trunk returning the final stage feature map (B, H, W, C)."""

    KMODELS_CONFIG = NEXTVIT_CONFIG
    KMODELS_WEIGHTS = NEXTVIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = NextViTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_nextvit_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 4, 10, 3),
        stem_chs=(64, 32, 64),
        head_dim=32,
        mix_block_ratio=0.75,
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="NextViTModel",
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
        features = _nextvit_features(
            x,
            depths=depths,
            stem_chs=stem_chs,
            head_dim=head_dim,
            mix_block_ratio=mix_block_ratio,
            sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features[-1], name=name, **kwargs)

        self.depths = list(depths)
        self.stem_chs = list(stem_chs)
        self.head_dim = head_dim
        self.mix_block_ratio = mix_block_ratio
        self.sr_ratios = list(sr_ratios)
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "stem_chs": self.stem_chs,
                "head_dim": self.head_dim,
                "mix_block_ratio": self.mix_block_ratio,
                "sr_ratios": self.sr_ratios,
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
class NextViTBackbone(BaseModel):
    """NextViT feature extractor. Returns ``[stem, s1..s4]`` (5 maps)."""

    KMODELS_CONFIG = NEXTVIT_CONFIG
    KMODELS_WEIGHTS = NEXTVIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = NextViTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_nextvit_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 4, 10, 3),
        stem_chs=(64, 32, 64),
        head_dim=32,
        mix_block_ratio=0.75,
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="NextViTBackbone",
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
        features = _nextvit_features(
            x,
            depths=depths,
            stem_chs=stem_chs,
            head_dim=head_dim,
            mix_block_ratio=mix_block_ratio,
            sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.depths = list(depths)
        self.stem_chs = list(stem_chs)
        self.head_dim = head_dim
        self.mix_block_ratio = mix_block_ratio
        self.sr_ratios = list(sr_ratios)
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "stem_chs": self.stem_chs,
                "head_dim": self.head_dim,
                "mix_block_ratio": self.mix_block_ratio,
                "sr_ratios": self.sr_ratios,
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
