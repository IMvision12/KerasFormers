import keras
from keras import layers, utils
from keras.src.applications import imagenet_utils

from kerasformers.base import BaseModel
from kerasformers.layers import ImageNormalizationLayer
from kerasformers.models.mobilevit.mobilevit_layers import (
    ImageToPatchesLayer,
    MultiHeadSelfAttention,
    PatchesToImageLayer,
)
from kerasformers.weight_utils import copy_weights_by_path_suffix

from .config import MOBILEVIT_MODEL_CONFIG, MOBILEVIT_WEIGHT_CONFIG
from .convert_mobilevit_torch_to_keras import transfer_mobilevit_weights


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """Snap a (possibly scaled) channel count to a multiple of ``divisor``.

    Args:
        v: Channel count to round (may be float-valued from a width multiplier).
        divisor: Multiple to snap to. Defaults to ``8``.
        min_value: Floor for the rounded value; defaults to ``divisor`` when ``None``.
        round_limit: If the rounded value is less than ``round_limit * v``, bump it
            up by one more ``divisor`` step. Defaults to ``0.9``.

    Returns:
        Integer channel count that is a multiple of ``divisor`` and at least
        ``min_value``.
    """
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
    """Inverted residual (MBConv) block as used in MobileNetV2 / MobileViT.

    Args:
        inputs: Input feature map.
        filters: Output channel count.
        channels_axis: Channel axis index (``-1`` for channels-last, ``-3`` for
            channels-first).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        strides: Spatial stride for the depthwise conv. Defaults to ``1``.
        expansion_ratio: Channel expansion factor for the hidden layer.
            Defaults to ``1.0``.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with ``filters`` channels and spatial size reduced by
        ``strides``.
    """
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
    """MobileViT transformer fusion block (local conv + global self-attention).

    Args:
        inputs: Input feature map.
        block_dims: Output channel count of the block.
        channels_axis: Channel axis index.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        attention_dims: Channel dimension used inside the transformer. If
            ``None``, defaults to ``make_divisible(inputs.shape[channels_axis])``.
        num_attention_blocks: Number of stacked transformer encoder blocks.
            Defaults to ``2``.
        patch_size: Side length of the square patches unfolded for self-attention.
            Defaults to ``8``.
        name: Prefix for layer names within this block.

    Returns:
        Output tensor with ``block_dims`` channels and the same spatial size as
        ``inputs``.
    """
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
    block_dims,
    expansion_ratio,
    attention_dims,
    data_format,
    channels_axis,
    return_stages=False,
):
    """MobileViT stem + 5 stages.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        initial_dims: Stem output channel count.
        block_dims: Per-stage output channel counts (length 5).
        expansion_ratio: Per-stage MBConv expansion ratios (length 5).
        attention_dims: Per-stage transformer attention dims (length 5). Entries
            may be ``None`` for stages that don't use a transformer block.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index.
        return_stages: If ``True``, return a list of the 5 per-stage feature
            maps instead of just the final one. Defaults to ``False``.

    Returns:
        Final stage feature map with ``block_dims[-1]`` channels at spatial
        resolution ``H/32`` when ``return_stages=False``. When
        ``return_stages=True``, a list of 5 per-stage feature maps.
    """
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

    stages = []
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

        stages.append(x)

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="kerasformers")
class MobileViTModel(BaseModel):
    """Instantiates the MobileViT backbone.

    MobileViT is a hybrid CNN-Transformer backbone designed for mobile
    inference: it interleaves MobileNetV2 inverted-residual (MBConv)
    blocks with MobileViT blocks that fold local self-attention over
    fixed-size patches, mixing convolutional locality with transformer
    global context. The network is organized as 5 stages of progressively
    reduced spatial resolution.

    Output is the last layer output before the classifier head:
    the final stage feature map ``(B, H, W, C)`` (channels-last) /
    ``(B, C, H, W)`` (channels-first) with ``block_dims[-1]`` channels at
    spatial resolution ``H/32``. :class:`MobileViTClassify` composes this
    model and appends the final 1x1 conv + GAP + Dense head.

    References:
    - [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of the
            5 per-stage feature maps. Defaults to `False`.
        initial_dims: Integer, stem output channel count.
            Defaults to `16`.
        head_dims: Integer, channel count of the 1x1 final conv used by
            the classifier head. Defaults to `640`.
        block_dims: List of integers, per-stage output channel counts
            (length 5). Defaults to `[32, 64, 96, 128, 160]`.
        expansion_ratio: List of floats, per-stage MBConv expansion
            ratios (length 5). Defaults to `[4.0, 4.0, 4.0, 4.0, 4.0]`.
        attention_dims: List, per-stage transformer attention dims
            (length 5). Entries may be `None` for stages without a
            transformer block. Defaults to `[None, None, 144, 192, 240]`.
        image_size: Integer, square input resolution. Used to validate
            the input shape. Defaults to `256`.
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
        name: String, the name of the model. Defaults to `"MobileViTModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = {
        variant: MOBILEVIT_MODEL_CONFIG[meta["model"]]
        for variant, meta in MOBILEVIT_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = MOBILEVIT_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = MobileViTClassify.from_weights(variant, skip_mismatch=skip_mismatch)
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
        as_backbone=False,
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
        x = mobilevit_backbone_feature(
            x,
            initial_dims=initial_dims,
            block_dims=block_dims,
            expansion_ratio=expansion_ratio,
            attention_dims=attention_dims,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

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
        self.as_backbone = as_backbone

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
                "as_backbone": self.as_backbone,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class MobileViTClassify(BaseModel):
    """Instantiates the MobileViT classifier.

    This classifier wraps a :class:`MobileViTModel` backbone and attaches
    a 1x1 final conv + GlobalAveragePooling2D + Dense head to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`MobileViTModel`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

    Args:
        initial_dims: Integer, stem output channel count.
            Defaults to `16`.
        head_dims: Integer, channel count of the 1x1 final conv used by
            the classifier head. Defaults to `640`.
        block_dims: List of integers, per-stage output channel counts
            (length 5). Defaults to `[32, 64, 96, 128, 160]`.
        expansion_ratio: List of floats, per-stage MBConv expansion
            ratios (length 5). Defaults to `[4.0, 4.0, 4.0, 4.0, 4.0]`.
        attention_dims: List, per-stage transformer attention dims
            (length 5). Entries may be `None` for stages without a
            transformer block. Defaults to `[None, None, 144, 192, 240]`.
        image_size: Integer, square input resolution. Used to validate
            the input shape. Defaults to `256`.
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
            named `f"{name}_backbone"`. Defaults to `"MobileViTClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = {
        variant: MOBILEVIT_MODEL_CONFIG[meta["model"]]
        for variant, meta in MOBILEVIT_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = MOBILEVIT_WEIGHT_CONFIG
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

        backbone = MobileViTModel(
            initial_dims=initial_dims,
            head_dims=head_dims,
            block_dims=block_dims,
            expansion_ratio=expansion_ratio,
            attention_dims=attention_dims,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.Conv2D(
            head_dims,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name="final_conv",
        )(backbone.output)
        x = layers.BatchNormalization(
            axis=channels_axis,
            momentum=0.9,
            epsilon=1e-5,
            name="final_batchnorm",
        )(x)
        x = layers.Activation("swish", name="final_act")(x)
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(x)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="predictions",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

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
