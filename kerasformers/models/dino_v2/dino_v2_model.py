import keras
from keras import layers, utils

from kerasformers.base import BaseModel
from kerasformers.layers import ImageNormalizationLayer
from kerasformers.models.vit.vit_model import vit_backbone_feature
from kerasformers.utils import standardize_input_shape

from .config import DINOV2_CONFIG, DINOV2_WEIGHTS


@keras.saving.register_keras_serializable(package="kerasformers")
class DinoV2Model(BaseModel):
    """DINOv2 Vision Transformer model.

    Standard ViT with LayerScale pretrained with the DINOv2
    self-supervised method.

    When ``as_backbone=False`` (default), returns the final
    LayerNorm-normalized token sequence ``(B, num_tokens, dim)`` (CLS at
    index 0). When ``as_backbone=True``, returns the list of
    intermediate feature maps from each transformer block (with the
    last LayerNorm-normalized), suitable for feeding into detection /
    segmentation / depth necks (e.g. Depth Anything, OwlViT).

    Reference:
        - `DINOv2: Learning Robust Visual Features without Supervision
          <https://arxiv.org/abs/2304.07193>`_

    Args:
        as_backbone: If ``True``, output the list of per-block
            intermediate features (last LayerNorm-normalized) for use as
            a backbone. If ``False`` (default), output only the final
            LayerNorm-normalized token sequence.
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
        input_image_shape: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()`` —
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = DINOV2_CONFIG
    BASE_WEIGHT_CONFIG = DINOV2_WEIGHTS
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
        from kerasformers.models.dino_v2.convert_dino_v2_hf_to_keras import (
            transfer_dino_v2_weights,
        )

        transfer_dino_v2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        as_backbone=False,
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
        input_image_shape=224,
        input_tensor=None,
        name="DinoV2Model",
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
            init_values=init_values,
            image_size=image_size,
            data_format=data_format,
            return_intermediates=True,
        )
        final_ln = layers.LayerNormalization(
            epsilon=1e-6, axis=-1, name="final_layernorm"
        )
        features[-1] = final_ln(features[-1])

        outputs = features if as_backbone else features[-1]
        super().__init__(inputs=img_input, outputs=outputs, name=name, **kwargs)

        self.as_backbone = as_backbone
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
        self.input_image_shape = input_image_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "as_backbone": self.as_backbone,
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
                "input_image_shape": self.input_image_shape,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
