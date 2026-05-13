"""DeiT and DeiT3 as thin :class:`ViT` subclasses (timm-ported)."""

import keras

from kmodels.models.vit.vit_model import ViT, ViTBackbone

from .config import DEIT_CONFIG, DEIT_WEIGHTS
from .convert_deit_torch_to_keras import transfer_deit_weights


@keras.saving.register_keras_serializable(package="kmodels")
class DeiT(ViT):
    """Data-efficient Image Transformer / DeiT3 classifier.

    Reference:
    - [DeiT](https://arxiv.org/abs/2012.12877)
    - [DeiT III](https://arxiv.org/abs/2204.07118)

    Construction:

    >>> DeiT.from_weights("deit3_base_patch16_224_fb_in22k_ft_in1k")
    >>> DeiT.from_weights("timm:timm/deit_tiny_distilled_patch16_224.fb_in1k")
    """

    KMODELS_CONFIG = DEIT_CONFIG
    KMODELS_WEIGHTS = DEIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_deit_weights(keras_model, state_dict)

    def __init__(self, name="DeiT", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class DeiTBackbone(ViTBackbone):
    """DeiT feature extractor (no classifier head). Returns final encoder tokens."""

    KMODELS_CONFIG = DEIT_CONFIG
    KMODELS_WEIGHTS = DEIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def _release_warm_start_cls(cls):
        return DeiT

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_deit_weights(keras_model, state_dict)

    def __init__(self, name="DeiTBackbone", **kwargs):
        super().__init__(name=name, **kwargs)
