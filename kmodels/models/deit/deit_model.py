"""DeiT and DeiT3 as thin :class:`ViTClassify` subclasses (timm-ported)."""

import keras
from keras import layers

from kmodels.models.vit.vit_model import ViTClassify, ViTModel
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import DEIT_MODEL_CONFIG, DEIT_WEIGHT_CONFIG
from .convert_deit_torch_to_keras import transfer_deit_weights


@keras.saving.register_keras_serializable(package="kmodels")
class DeiTModel(ViTModel):
    """DeiT backbone — thin :class:`ViTModel` subclass that loads DeiT/DeiT3 timm weights.

    Returns the final-LN normalized token sequence ``(B, num_tokens, dim)``.
    The first 1 (or 2 if distillation) tokens are class/distillation tokens;
    the rest are spatial patch tokens.
    """

    BASE_MODEL_CONFIG = {
        v: DEIT_MODEL_CONFIG[m["model"]] for v, m in DEIT_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = DEIT_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = DeiTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_deit_weights(keras_model, state_dict)

    def __init__(self, as_backbone=False, name="DeiTModel", **kwargs):
        super().__init__(as_backbone=as_backbone, name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class DeiTClassify(ViTClassify):
    """Data-efficient Image Transformer / DeiT3 classifier.

    Reference:
    - [DeiT](https://arxiv.org/abs/2012.12877)
    - [DeiT III](https://arxiv.org/abs/2204.07118)

    Construction:

    >>> DeiTClassify.from_weights("deit3_base_patch16_224_fb_in22k_ft_in1k")
    >>> DeiTClassify.from_weights("timm:timm/deit_tiny_distilled_patch16_224.fb_in1k")
    """

    BASE_MODEL_CONFIG = {
        v: DEIT_MODEL_CONFIG[m["model"]] for v, m in DEIT_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = DEIT_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_deit_weights(keras_model, state_dict)

    def __init__(
        self,
        patch_size=16,
        dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        no_embed_class=False,
        use_distillation=False,
        init_values=None,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="DeiTClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        backbone = DeiTModel(
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            no_embed_class=no_embed_class,
            use_distillation=use_distillation,
            init_values=init_values,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = backbone.output
        if use_distillation:
            cls_token = layers.Lambda(lambda v: v[:, 0], name="ExtractClsToken")(x)
            dist_token = layers.Lambda(lambda v: v[:, 1], name="ExtractDistToken")(x)
            cls_token = layers.Dropout(drop_rate)(cls_token)
            dist_token = layers.Dropout(drop_rate)(dist_token)
            cls_head = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(cls_token)
            dist_head = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions_dist",
            )(dist_token)
            out = (cls_head + dist_head) / 2
        else:
            tok = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
            tok = layers.Dropout(drop_rate)(tok)
            out = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
            )(tok)

        super(ViTClassify, self).__init__(
            inputs=backbone.input, outputs=out, name=name, **kwargs
        )

        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.no_embed_class = no_embed_class
        self.use_distillation = use_distillation
        self.init_values = init_values
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
