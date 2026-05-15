"""FlexiViT as a thin :class:`ViTClassify` subclass (timm-ported)."""

import keras
from keras import layers

from kmodels.models.vit.convert_vit_torch_to_keras import transfer_vit_weights
from kmodels.models.vit.vit_model import ViTClassify, ViTModel
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import FLEXIVIT_CONFIG, FLEXIVIT_WEIGHTS


@keras.saving.register_keras_serializable(package="kmodels")
class FlexiViTModel(ViTModel):
    """FlexiViT backbone — thin :class:`ViTModel` subclass that loads FlexiViT timm weights.

    Returns the final-LN normalized token sequence ``(B, num_tokens, dim)``.
    The first token is the class token; the rest are spatial patch tokens.
    """

    KMODELS_CONFIG = FLEXIVIT_CONFIG
    KMODELS_WEIGHTS = FLEXIVIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = FlexiViTClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_vit_weights(keras_model, state_dict)

    def __init__(self, name="FlexiViTModel", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class FlexiViTClassify(ViTClassify):
    """FlexiViT classifier (no_embed_class=True for flexible patch sizes).

    Reference:
    - [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013)

    Construction:

    >>> FlexiViTClassify.from_weights("flexivit_base_1200ep_in1k")
    >>> FlexiViTClassify.from_weights("timm:timm/flexivit_base.1200ep_in1k")
    """

    KMODELS_CONFIG = FLEXIVIT_CONFIG
    KMODELS_WEIGHTS = FLEXIVIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_vit_weights(keras_model, state_dict)

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
        name="FlexiViTClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        backbone = FlexiViTModel(
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
