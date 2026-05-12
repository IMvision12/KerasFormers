import keras

from kmodels.base import BaseModel
from kmodels.models.resnet.resnet_model import ResNet, bottleneck_block
from kmodels.models.vit.vit_model import VisionTransformer

from .config import (
    DINO_RESNET_CONFIG,
    DINO_RESNET_WEIGHTS,
    DINO_VIT_CONFIG,
    DINO_VIT_WEIGHTS,
)


@keras.saving.register_keras_serializable(package="kmodels")
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
        input_shape: Image input shape excluding batch dim. Defaults
            to ``(224, 224, 3)``.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    KMODELS_CONFIG = DINO_VIT_CONFIG
    KMODELS_WEIGHTS = DINO_VIT_WEIGHTS
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
        input_shape=None,
        input_tensor=None,
        name="DinoViTBackbone",
        **kwargs,
    ):
        if input_shape is None and input_tensor is None:
            input_shape = (224, 224, 3)

        base = VisionTransformer(
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            include_top=False,
            as_backbone=True,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            weights=None,
            input_tensor=input_tensor,
            input_shape=input_shape,
            name=f"{name}_vit",
        )

        super().__init__(inputs=base.input, outputs=base.output, name=name, **kwargs)

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
        self._input_shape_val = input_shape
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
                "input_shape": self._input_shape_val,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
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
        input_shape: Image input shape excluding batch dim.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    KMODELS_CONFIG = DINO_RESNET_CONFIG
    KMODELS_WEIGHTS = DINO_RESNET_WEIGHTS
    HF_MODEL_TYPE = None

    def __init__(
        self,
        block_repeats=None,
        filters=None,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="DinoResNetBackbone",
        **kwargs,
    ):
        if block_repeats is None:
            block_repeats = [3, 4, 6, 3]
        if filters is None:
            filters = [64, 128, 256, 512]

        base = ResNet(
            block_fn=bottleneck_block,
            block_repeats=block_repeats,
            filters=filters,
            include_top=False,
            as_backbone=True,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            weights=None,
            input_tensor=input_tensor,
            input_shape=input_shape,
            name=f"{name}_resnet",
        )

        super().__init__(inputs=base.input, outputs=base.output, name=name, **kwargs)

        self.block_repeats = list(block_repeats)
        self.filters = list(filters)
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "block_repeats": self.block_repeats,
                "filters": self.filters,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_shape": self._input_shape_val,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
