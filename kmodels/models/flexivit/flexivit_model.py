"""FlexiViT as a thin :class:`ViT` subclass (timm-ported)."""

import keras

from kmodels.models.vit.convert_vit_torch_to_keras import transfer_vit_weights
from kmodels.models.vit.vit_model import ViT, ViTBackbone

from .config import FLEXIVIT_CONFIG, FLEXIVIT_WEIGHTS


@keras.saving.register_keras_serializable(package="kmodels")
class FlexiViT(ViT):
    """FlexiViT classifier (no_embed_class=True for flexible patch sizes).

    Reference:
    - [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013)

    Construction:

    >>> FlexiViT.from_weights("flexivit_base_1200ep_in1k")
    >>> FlexiViT.from_weights("timm:timm/flexivit_base.1200ep_in1k")
    """

    KMODELS_CONFIG = FLEXIVIT_CONFIG
    KMODELS_WEIGHTS = FLEXIVIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_vit_weights(keras_model, state_dict)

    def __init__(self, name="FlexiViT", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class FlexiViTBackbone(ViTBackbone):
    """FlexiViT feature extractor (no classifier head). Returns final encoder tokens."""

    KMODELS_CONFIG = FLEXIVIT_CONFIG
    KMODELS_WEIGHTS = FLEXIVIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        return FlexiViT

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_vit_weights(keras_model, state_dict)

    def __init__(self, name="FlexiViTBackbone", **kwargs):
        super().__init__(name=name, **kwargs)
