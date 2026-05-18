import keras
from keras import layers, utils

from kerasformers.base import BaseModel
from kerasformers.layers import ImageNormalizationLayer
from kerasformers.models.resnet.resnet_model import (
    bottleneck_block,
    resnet_backbone_feature,
)
from kerasformers.models.vit.vit_model import vit_backbone_feature
from kerasformers.utils import standardize_input_shape

from .config import (
    DINO_RESNET_CONFIG,
    DINO_RESNET_WEIGHTS,
    DINO_VIT_CONFIG,
    DINO_VIT_WEIGHTS,
)


@keras.saving.register_keras_serializable(package="kerasformers")
class DinoViTBackbone(BaseModel):
    """DINO Vision Transformer backbone.

    Standard ViT pretrained with the DINO self-supervised method.
    Returns the list of intermediate feature maps from each
    transformer block, suitable for feeding into detection /
    segmentation / depth necks.

    Reference:
        - `Emerging Properties in Self-Supervised Vision Transformers
          <https://arxiv.org/abs/2104.14294>`_

    Args:
        patch_size: ViT patch size (8 or 16).
        dim: Hidden dimension.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        mlp_ratio: MLP expansion ratio. Defaults to ``4.0``.
        qkv_bias: Whether to use bias in QKV projections. Defaults to
            ``True``.
        qk_norm: Whether to use QK normalization. Defaults to ``False``.
        drop_rate: Dropout rate. Defaults to ``0.0``.
        attn_drop_rate: Attention dropout rate. Defaults to ``0.0``.
        include_normalization: Whether to prepend
            :class:`ImageNormalizationLayer`.
        normalization_mode: Normalization preset.
        input_image_shape: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()`` —
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = DINO_VIT_CONFIG
    BASE_WEIGHT_CONFIG = DINO_VIT_WEIGHTS
    HF_MODEL_TYPE = None

    def __init__(
        self,
        patch_size=16,
        dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        include_normalization=True,
        normalization_mode="imagenet",
        input_image_shape=224,
        input_tensor=None,
        name="DinoViTBackbone",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        input_image_shape = standardize_input_shape(input_image_shape, data_format)
        image_size = (
            input_image_shape[0]
            if data_format == "channels_last"
            else input_image_shape[1]
        )

        if input_tensor is None:
            img_input = layers.Input(shape=input_image_shape)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_image_shape)
        else:
            img_input = input_tensor

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(img_input)
            if include_normalization
            else img_input
        )
        features = vit_backbone_feature(
            x,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            no_embed_class=False,
            use_distillation=False,
            init_values=None,
            image_size=image_size,
            data_format=data_format,
            return_intermediates=True,
        )
        final_ln = layers.LayerNormalization(
            epsilon=1e-6, axis=-1, name="final_layernorm"
        )
        features[-1] = final_ln(features[-1])

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_image_shape = input_image_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "dim": self.dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "qk_norm": self.qk_norm,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_image_shape": self.input_image_shape,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class DinoResNetBackbone(BaseModel):
    """DINO ResNet backbone.

    ResNet-50 pretrained with the DINO self-supervised method. Returns
    the list of intermediate feature maps from each ResNet stage.

    Reference:
        - `Emerging Properties in Self-Supervised Vision Transformers
          <https://arxiv.org/abs/2104.14294>`_

    Args:
        block_repeats: Per-stage block counts.
        filters: Per-stage filter counts.
        include_normalization: Whether to prepend
            :class:`ImageNormalizationLayer`.
        normalization_mode: Normalization preset.
        input_image_shape: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()`` —
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = DINO_RESNET_CONFIG
    BASE_WEIGHT_CONFIG = DINO_RESNET_WEIGHTS
    HF_MODEL_TYPE = None

    def __init__(
        self,
        block_repeats=None,
        filters=None,
        include_normalization=True,
        normalization_mode="imagenet",
        input_image_shape=224,
        input_tensor=None,
        name="DinoResNetBackbone",
        **kwargs,
    ):
        if block_repeats is None:
            block_repeats = [3, 4, 6, 3]
        if filters is None:
            filters = [64, 128, 256, 512]

        data_format = keras.config.image_data_format()
        input_image_shape = standardize_input_shape(input_image_shape, data_format)
        channels_axis = -1 if data_format == "channels_last" else 1

        if input_tensor is None:
            img_input = layers.Input(shape=input_image_shape)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_image_shape)
        else:
            img_input = input_tensor

        x = (
            ImageNormalizationLayer(mode=normalization_mode)(img_input)
            if include_normalization
            else img_input
        )
        features = resnet_backbone_feature(
            x,
            block_fn=bottleneck_block,
            block_repeats=block_repeats,
            filters=filters,
            channels_axis=channels_axis,
            data_format=data_format,
            groups=1,
            senet=False,
            width_factor=1,
            return_stages=True,
        )

        super().__init__(inputs=img_input, outputs=features, name=name, **kwargs)

        self.block_repeats = list(block_repeats)
        self.filters = list(filters)
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_image_shape = input_image_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "block_repeats": self.block_repeats,
                "filters": self.filters,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_image_shape": self.input_image_shape,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
