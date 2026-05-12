import keras
from keras import layers

from kmodels.base import BaseModel
from kmodels.models.dino_v2.convert_dino_v2_hf_to_keras import transfer_dino_v2_weights
from kmodels.models.vit.vit_model import VisionTransformer

from .config import DINOV2_CONFIG, DINOV2_WEIGHTS


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV2Backbone(BaseModel):
    """DINOv2 Vision Transformer backbone.

    Standard ViT with LayerScale pretrained with the DINOv2
    self-supervised method. Returns the list of intermediate feature
    maps from each transformer block, suitable for feeding into
    detection / segmentation / depth necks (e.g. Depth Anything,
    OwlViT).

    Reference:
        - `DINOv2: Learning Robust Visual Features without Supervision
          <https://arxiv.org/abs/2304.07193>`_

    Args:
        patch_size: ViT patch size. DINOv2 uses 14.
        dim: Hidden dimension.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads per layer.
        mlp_ratio: MLP expansion ratio. Defaults to ``4.0``.
        qkv_bias: Whether to use bias in QKV projections. Defaults to
            ``True``.
        qk_norm: Whether to use QK normalization. Defaults to ``False``.
        drop_rate: Dropout rate. Defaults to ``0.0``.
        attn_drop_rate: Attention dropout rate. Defaults to ``0.0``.
        init_values: LayerScale init value. Defaults to ``1.0``.
        include_normalization: Whether to prepend
            :class:`ImageNormalizationLayer`.
        normalization_mode: Normalization preset.
        input_shape: Image input shape excluding batch dim. Defaults
            to ``(224, 224, 3)``.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    KMODELS_CONFIG = DINOV2_CONFIG
    KMODELS_WEIGHTS = DINOV2_WEIGHTS
    HF_MODEL_TYPE = "dinov2"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "patch_size": hf_config.get("patch_size", 14),
            "dim": hf_config["hidden_size"],
            "depth": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "mlp_ratio": hf_config.get("mlp_ratio", 4.0),
            "init_values": hf_config.get("layerscale_value", 1.0),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        transfer_dino_v2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        patch_size=14,
        dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        init_values=1.0,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        name="DinoV2Backbone",
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
            init_values=init_values,
            include_top=False,
            as_backbone=True,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            weights=None,
            input_tensor=input_tensor,
            input_shape=input_shape,
            name=f"{name}_vit",
        )
        features = list(base.output)
        final_ln = layers.LayerNormalization(
            epsilon=1e-6, axis=-1, name="final_layernorm"
        )
        features[-1] = final_ln(features[-1])

        super().__init__(inputs=base.input, outputs=features, name=name, **kwargs)

        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.init_values = init_values
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
                "init_values": self.init_values,
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
