"""FlexiViT as a thin :class:`ViTClassify` subclass (timm-ported)."""

import keras

from kmodels.models.vit.convert_vit_torch_to_keras import transfer_vit_weights
from kmodels.models.vit.vit_model import ViTBackbone, ViTClassify, ViTModel
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import FLEXIVIT_CONFIG, FLEXIVIT_WEIGHTS


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

    def __init__(self, name="FlexiViTClassify", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class FlexiViTBackbone(ViTBackbone):
    """FlexiViT feature extractor (no classifier head). Returns final encoder tokens."""

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

    def __init__(self, name="FlexiViTBackbone", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class FlexiViTModel(ViTModel):
    """FlexiViT trunk returning the final feature as a 4D map."""

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
