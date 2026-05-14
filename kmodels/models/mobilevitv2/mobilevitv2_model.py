import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kmodels.base import BaseModel
from kmodels.layers import ImageNormalizationLayer
from kmodels.models.mobilevit.mobilevit_layers import (
    ImageToPatchesLayer,
    PatchesToImageLayer,
)
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import MOBILEVITV2_CONFIG, MOBILEVITV2_WEIGHTS
from .convert_mobilevitv2_torch_to_keras import transfer_mobilevitv2_weights


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
    expansion_ratio=2.0,
    name="inverted_residual_block",
):
    """MobileViTV2 inverted residual block."""
    residual_connection = (strides == 1) and (inputs.shape[channels_axis] == filters)

    x = layers.Conv2D(
        make_divisible(inputs.shape[channels_axis] * expansion_ratio),
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
            padding=1,
            data_format=data_format,
            name=f"{name}_ir_zeropadding",
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
        x = layers.Add(name=f"{name}_ir_add")([x, inputs])

    return x


def linear_self_attention(
    inputs, dim, data_format, use_bias=True, name="linear_self_attention"
):
    """Linear self-attention block used by MobileViTV2."""
    num_patch_axis = -2 if data_format == "channels_last" else -1

    x = layers.Conv2D(1 + (2 * dim), 1, use_bias=use_bias, name=f"{name}_attn_conv_1")(
        inputs
    )

    if data_format == "channels_last":
        query = x[..., :1]
        key = x[..., 1 : dim + 1]
        value = x[..., dim + 1 :]
    else:
        query = x[:, :1]
        key = x[:, 1 : dim + 1]
        value = x[:, dim + 1 :]

    context_scores = layers.Softmax(axis=num_patch_axis, name=f"{name}_attn_softmax")(
        query
    )
    context_vector = layers.Multiply(name=f"{name}_attn_multiply_1")(
        [key, context_scores]
    )
    context_vector = keras.ops.sum(context_vector, axis=num_patch_axis, keepdims=True)

    out = layers.ReLU(name=f"{name}_attn_relu")(value)
    out = layers.Multiply(name=f"{name}_attn_multiply_2")([out, context_vector])
    out = layers.Conv2D(dim, 1, use_bias=use_bias, name=f"{name}_attn_conv_2")(out)

    return out


def mobilevitv2_block(
    inputs,
    block_dims,
    channels_axis,
    data_format,
    kernel_size=3,
    expansion_ratio=2.0,
    transformer_dim=None,
    transformer_depth=2,
    patch_size=2,
    name="mobilevitv2_block",
):
    """MobileViTV2 transformer block with linear self-attention."""
    transformer_dim = transformer_dim or make_divisible(
        inputs.shape[channels_axis] * expansion_ratio
    )

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv2_dwconv",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv2_batchnorm_1",
    )(x)
    x = layers.Activation("swish", name=f"{name}_mc2_act_1")(x)

    x = layers.Conv2D(
        transformer_dim,
        1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv2_conv_1",
    )(x)

    if data_format == "channels_first":
        h, w = x.shape[-2], x.shape[-1]
    else:
        h, w = x.shape[-3], x.shape[-2]

    unfold_layer = ImageToPatchesLayer(patch_size)
    x = unfold_layer(x)
    resize = unfold_layer.resize

    for i in range(transformer_depth):
        residual = x
        x = layers.GroupNormalization(
            1,
            axis=channels_axis,
            epsilon=1e-5,
            name=f"{name}_transformer_{i}_groupnorm_1",
        )(x)
        x = linear_self_attention(
            x,
            transformer_dim,
            data_format,
            use_bias=True,
            name=f"{name}_transformer_{i}",
        )
        x = layers.Add(name=f"{name}_transformer_{i}_add_1")([residual, x])

        residual = x
        x = layers.GroupNormalization(
            1,
            axis=channels_axis,
            epsilon=1e-5,
            name=f"{name}_transformer_{i}_groupnorm_2",
        )(x)
        mlp_hidden_dim = int(transformer_dim * 2.0)

        x = layers.Conv2D(
            mlp_hidden_dim,
            1,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_conv_1",
        )(x)
        x = layers.Activation("swish", name=f"{name}_transformer_{i}_mlp_act")(x)
        x = layers.Conv2D(
            transformer_dim,
            1,
            use_bias=True,
            name=f"{name}_transformer_{i}_mlp_conv_2",
        )(x)
        x = layers.Add(name=f"{name}_transformer_{i}_add_2")([residual, x])

    x = layers.GroupNormalization(
        1,
        axis=channels_axis,
        epsilon=1e-5,
        name=f"{name}_groupnorm",
    )(x)

    fold_layer = PatchesToImageLayer(patch_size)
    x = fold_layer(x, original_size=(h, w), resize=resize)

    x = layers.Conv2D(
        block_dims,
        1,
        strides=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_mv2_proj_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=0.9,
        epsilon=1e-5,
        name=f"{name}_mv2_proj_batchnorm",
    )(x)

    return x


def mobilevitv2_backbone_feature(
    inputs,
    *,
    multiplier,
    data_format,
    channels_axis,
):
    """MobileViTV2 stem + 5 stages.

    Returns ``[stem, stage0, stage1, stage2, stage3, stage4]``.
    """
    features = []

    x = layers.ZeroPadding2D(padding=1, data_format=data_format)(inputs)
    x = layers.Conv2D(
        int(32 * multiplier),
        3,
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

    for stage in range(5):
        channels = int(([64, 128, 256, 384, 512][stage]) * multiplier)
        stride = 1 if stage == 0 else 2

        x = inverted_residual_block(
            x,
            channels,
            channels_axis,
            data_format,
            strides=stride,
            expansion_ratio=2.0,
            name=f"stages_{stage}_0",
        )

        if stage <= 1:
            if stage == 1:
                x = inverted_residual_block(
                    x,
                    channels,
                    channels_axis,
                    data_format,
                    strides=1,
                    expansion_ratio=2.0,
                    name=f"stages_{stage}_1",
                )
        else:
            x = mobilevitv2_block(
                x,
                channels,
                channels_axis,
                data_format,
                kernel_size=3,
                expansion_ratio=0.5,
                transformer_depth=[2, 4, 3][stage - 2],
                patch_size=2,
                name=f"stages_{stage}_1",
            )

        features.append(x)

    return features


@keras.saving.register_keras_serializable(package="kmodels")
class MobileViTV2Classify(BaseModel):
    """MobileViTV2 classifier (timm-ported).

    Reference:
    - [Separable Self-attention for Mobile Vision
      Transformers](https://arxiv.org/abs/2206.02680)

    Construction:

    >>> MobileViTV2.from_weights("mobilevitv2_100_cvnets_in1k")
    >>> MobileViTV2.from_weights("timm:timm/mobilevitv2_100.cvnets_in1k")
    """

    KMODELS_CONFIG = MOBILEVITV2_CONFIG
    KMODELS_WEIGHTS = MOBILEVITV2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mobilevitv2_weights(keras_model, state_dict)

    def __init__(
        self,
        multiplier=1.0,
        image_size=256,
        include_normalization=True,
        normalization_mode="zero_to_one",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="MobileViTV2Classify",
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
        features = mobilevitv2_backbone_feature(
            x,
            multiplier=multiplier,
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

        self.multiplier = multiplier
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
                "multiplier": self.multiplier,
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
class MobileViTV2Model(BaseModel):
    """MobileViTV2 trunk returning the final stage feature map."""

    KMODELS_CONFIG = MOBILEVITV2_CONFIG
    KMODELS_WEIGHTS = MOBILEVITV2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = MobileViTV2Classify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mobilevitv2_weights(keras_model, state_dict)

    def __init__(
        self,
        multiplier=1.0,
        image_size=256,
        include_normalization=True,
        normalization_mode="zero_to_one",
        input_shape=None,
        input_tensor=None,
        name="MobileViTV2Model",
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
        features = mobilevitv2_backbone_feature(
            x,
            multiplier=multiplier,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features[-1], name=name, **kwargs)

        self.multiplier = multiplier
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "multiplier": self.multiplier,
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
class MobileViTV2Backbone(BaseModel):
    """MobileViTV2 feature extractor (no classifier head)."""

    KMODELS_CONFIG = MOBILEVITV2_CONFIG
    KMODELS_WEIGHTS = MOBILEVITV2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = MobileViTV2Classify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_mobilevitv2_weights(keras_model, state_dict)

    def __init__(
        self,
        multiplier=1.0,
        image_size=256,
        include_normalization=True,
        normalization_mode="zero_to_one",
        input_shape=None,
        input_tensor=None,
        name="MobileViTV2Backbone",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        # require_flatten=True keeps a concrete H/W in the input spec which
        # the MobileViTV2 patch-fold layer needs at graph-build time.
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
        features = mobilevitv2_backbone_feature(
            x,
            multiplier=multiplier,
            data_format=data_format,
            channels_axis=channels_axis,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.multiplier = multiplier
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "multiplier": self.multiplier,
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
