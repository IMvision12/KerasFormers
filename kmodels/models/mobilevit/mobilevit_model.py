import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.models.mobilevit.mobilevit_layers import (
    ImageToPatchesLayer,
    MultiHeadSelfAttention,
    PatchesToImageLayer,
)
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import MOBILEVIT_CONFIG, MOBILEVIT_WEIGHTS
from .convert_mobilevit_torch_to_keras import transfer_mobilevit_weights


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """Snap a (possibly scaled) channel count to a multiple of ``divisor``."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def inverted_residual_block(
    inputs,
    filters,
    channels_axis,
    data_format,
    strides=1,
    expansion_ratio=1.0,
    name: str = "inverted_residual_block",
):
    """Creates an inverted residual block as used in MobileNetV2 / MobileViT."""
    residual_connection = (strides == 1) and (inputs.shape[channels_axis] == filters)

    x = layers.Conv2D(
        filters=make_divisible(inputs.shape[channels_axis] * expansion_ratio),
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_ir_conv_1",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_ir_batchnorm_1",
    )(x)
    x = layers.Activation("swish", name=f"{name}_ir_act_1")(x)

    if strides > 1:
        x = layers.ZeroPadding2D(
            padding=(1, 1),
            data_format=data_format,
            name=f"{name}_zeropadding",
        )(x)
        padding = "valid"
    else:
        padding = "same"

    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_ir_dwconv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_ir_batchnorm_2",
    )(x)
    x = layers.Activation("swish", name=f"{name}_ir_act_2")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_ir_conv_2",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_ir_batchnorm_3",
    )(x)

    if residual_connection:
        x = layers.Add(name=f"{name}_add")([x, inputs])

    return x


def mobilevit_block(
    inputs,
    block_dims,
    channels_axis,
    data_format,
    attention_dims=None,
    num_attention_blocks=2,
    patch_size=8,
    name="mobilevit_transformer_block",
):
    """MobileViT transformer fusion block."""
    if attention_dims is None:
        attention_dims = make_divisible(inputs.shape[channels_axis])

    x = inputs

    x = layers.Conv2D(
        inputs.shape[channels_axis],
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv_conv_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, momentum=0.9, epsilon=1e-5, name=f"{name}_mv_batchnorm_1"
    )(x)
    x = layers.Activation("swish", name=f"{name}_mv_act_1")(x)

    x = layers.Conv2D(attention_dims, 1, use_bias=False, name=f"{name}_mv_conv_2")(x)

    if data_format == "channels_first":
        h, w = x.shape[-2], x.shape[-1]
    else:
        h, w = x.shape[-3], x.shape[-2]

    unfold_layer = ImageToPatchesLayer(patch_size)
    x = unfold_layer(x)
    resize = unfold_layer.resize

    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1))(x)

    for i in range(num_attention_blocks):
        residual_1 = x
        x = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_transformer_{i}_layernorm_1"
        )(x)
        x = MultiHeadSelfAttention(
            attention_dims,
            num_heads=4,
            qkv_bias=True,
            block_prefix=f"{name}_transformer_{i}",
        )(x)
        x = layers.Add(name=f"{name}_transformer_{i}_add_1")([residual_1, x])

        residual_2 = x
        x = layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_transformer_{i}_layernorm_2"
        )(x)
        mlp_hidden_dim = int(attention_dims * 2.0)
        x = layers.Dense(
            mlp_hidden_dim,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_fc1",
        )(x)
        x = layers.Activation("swish", name=f"{name}_transformer_{i}_mlp_act")(x)
        x = layers.Dropout(0.0, name=f"{name}_transformer_{i}_mlp_drop_1")(x)
        x = layers.Dense(
            attention_dims,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_fc2",
        )(x)
        x = layers.Dropout(0.0, name=f"{name}_transformer_{i}_mlp_drop_2")(x)
        x = layers.Add(name=f"{name}_transformer_{i}_add_2")([residual_2, x])

    x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name=f"{name}_layernorm")(x)

    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2))(x)

    fold_layer = PatchesToImageLayer(patch_size)
    x = fold_layer(x, original_size=(h, w), resize=resize)

    x = layers.Conv2D(
        block_dims,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv_conv_3",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv_batchnorm_2",
    )(x)
    x = layers.Activation("swish", name=f"{name}_mv_act_2")(x)

    x = layers.Concatenate(axis=channels_axis, name=f"{name}_concat")([inputs, x])

    x = layers.Conv2D(
        block_dims,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv_conv_4",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv_batchnorm_3",
    )(x)
    x = layers.Activation("swish", name=f"{name}_mv_act_3")(x)

    return x


def mobilevit_backbone_feature(
    inputs,
    *,
    initial_dims,
    head_dims,
    block_dims,
    expansion_ratio,
    attention_dims,
    data_format,
    channels_axis,
):
    """MobileViT stem + 5 stages + final 1x1 conv.

    Returns ``[stem, stage0, stage1, stage2, stage3, stage4, final_conv]``.
    """
    features = []

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), data_format=data_format)(inputs)
    x = layers.Conv2D(
        initial_dims,
        kernel_size=3,
        strides=2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name="stem_batchnorm",
    )(x)
    x = layers.Activation("swish", name="stem_act")(x)
    features.append(x)

    for i in range(5):
        x = inverted_residual_block(
            x,
            filters=block_dims[i],
            channels_axis=channels_axis,
            data_format=data_format,
            strides=2 if i > 0 else 1,
            expansion_ratio=expansion_ratio[i],
            name=f"stages_{i}_0",
        )

        if i == 1:
            x = inverted_residual_block(
                x,
                filters=block_dims[i],
                channels_axis=channels_axis,
                data_format=data_format,
                strides=1,
                expansion_ratio=expansion_ratio[i],
                name=f"stages_{i}_1",
            )
            x = inverted_residual_block(
                x,
                filters=block_dims[i],
                channels_axis=channels_axis,
                data_format=data_format,
                strides=1,
                expansion_ratio=expansion_ratio[i],
                name=f"stages_{i}_2",
            )

        if i >= 2:
            x = mobilevit_block(
                x,
                block_dims=block_dims[i],
                channels_axis=channels_axis,
                data_format=data_format,
                attention_dims=attention_dims[i],
                num_attention_blocks=2 if i == 2 else 4 if i == 3 else 3,
                patch_size=2,
                name=f"stages_{i}_1",
            )

        features.append(x)

    x = layers.Conv2D(
        head_dims,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name="final_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name="final_batchnorm",
    )(x)
    x = layers.Activation("swish", name="final_act")(x)
    features.append(x)

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class MobileViTClassify(BaseModel):
    """MobileViT classifier (timm-ported).

    Reference:
    - [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision
      Transformer](https://arxiv.org/abs/2110.02178)

    Construction:

    >>> MobileViT.from_weights("mobilevit_s_cvnets_in1k")
    >>> MobileViT.from_weights("timm:timm/mobilevit_s.cvnets_in1k")
    """

    KMODELS_CONFIG = MOBILEVIT_CONFIG
    KMODELS_WEIGHTS = MOBILEVIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mobilevit_weights(keras_model, state_dict)

    def __init__(
        self,
        initial_dims: int = 16,
        head_dims: int = 640,
        block_dims: list = [32, 64, 96, 128, 160],
        expansion_ratio: list = [4.0, 4.0, 4.0, 4.0, 4.0],
        attention_dims: list = [None, None, 144, 192, 240],
        image_size=256,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="MobileViTClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

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
        features = mobilevit_backbone_feature(
            x,
            initial_dims=initial_dims,
            head_dims=head_dims,
            block_dims=block_dims,
            expansion_ratio=expansion_ratio,
            attention_dims=attention_dims,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            features[-1]
        )
        x = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.initial_dims = initial_dims
        self.head_dims = head_dims
        self.block_dims = block_dims
        self.expansion_ratio = expansion_ratio
        self.attention_dims = attention_dims
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
                "initial_dims": self.initial_dims,
                "head_dims": self.head_dims,
                "block_dims": self.block_dims,
                "expansion_ratio": self.expansion_ratio,
                "attention_dims": self.attention_dims,
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
class MobileViTModel(BaseModel):
    """MobileViT trunk returning the final stage feature map.

    Output shape: ``(B, H, W, C)`` — the final 1x1-projected feature map,
    unpooled and head-free.
    """

    KMODELS_CONFIG = MOBILEVIT_CONFIG
    KMODELS_WEIGHTS = MOBILEVIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = MobileViTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mobilevit_weights(keras_model, state_dict)

    def __init__(
        self,
        initial_dims: int = 16,
        head_dims: int = 640,
        block_dims: list = [32, 64, 96, 128, 160],
        expansion_ratio: list = [4.0, 4.0, 4.0, 4.0, 4.0],
        attention_dims: list = [None, None, 144, 192, 240],
        image_size=256,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="MobileViTModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

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
        features = mobilevit_backbone_feature(
            x,
            initial_dims=initial_dims,
            head_dims=head_dims,
            block_dims=block_dims,
            expansion_ratio=expansion_ratio,
            attention_dims=attention_dims,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features[-1], name=name, **kwargs)

        self.initial_dims = initial_dims
        self.head_dims = head_dims
        self.block_dims = block_dims
        self.expansion_ratio = expansion_ratio
        self.attention_dims = attention_dims
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initial_dims": self.initial_dims,
                "head_dims": self.head_dims,
                "block_dims": self.block_dims,
                "expansion_ratio": self.expansion_ratio,
                "attention_dims": self.attention_dims,
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
class MobileViTBackbone(BaseModel):
    """MobileViT feature extractor (no classifier head)."""

    KMODELS_CONFIG = MOBILEVIT_CONFIG
    KMODELS_WEIGHTS = MOBILEVIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = MobileViTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mobilevit_weights(keras_model, state_dict)

    def __init__(
        self,
        initial_dims: int = 16,
        head_dims: int = 640,
        block_dims: list = [32, 64, 96, 128, 160],
        expansion_ratio: list = [4.0, 4.0, 4.0, 4.0, 4.0],
        attention_dims: list = [None, None, 144, 192, 240],
        image_size=256,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="MobileViTBackbone",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        # require_flatten=True keeps a concrete H/W in the input spec which
        # the MobileViT patch-fold layer needs at graph-build time.
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
        features = mobilevit_backbone_feature(
            x,
            initial_dims=initial_dims,
            head_dims=head_dims,
            block_dims=block_dims,
            expansion_ratio=expansion_ratio,
            attention_dims=attention_dims,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.initial_dims = initial_dims
        self.head_dims = head_dims
        self.block_dims = block_dims
        self.expansion_ratio = expansion_ratio
        self.attention_dims = attention_dims
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initial_dims": self.initial_dims,
                "head_dims": self.head_dims,
                "block_dims": self.block_dims,
                "expansion_ratio": self.expansion_ratio,
                "attention_dims": self.attention_dims,
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
