"""DeiT and DeiT3 as thin :class:`ViTClassify` subclasses (timm-ported)."""

import keras

from kmodels.models.vit.vit_model import ViTBackbone, ViTClassify, ViTModel
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import DEIT_CONFIG, DEIT_WEIGHTS
from .convert_deit_torch_to_keras import transfer_deit_weights


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

    KMODELS_CONFIG = DEIT_CONFIG
    KMODELS_WEIGHTS = DEIT_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_deit_weights(keras_model, state_dict)

    def __init__(self, name="DeiTClassify", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class DeiTBackbone(ViTBackbone):
    """DeiT feature extractor (no classifier head). Returns final encoder tokens."""

    KMODELS_CONFIG = DEIT_CONFIG
    KMODELS_WEIGHTS = DEIT_WEIGHTS
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

    def __init__(self, name="DeiTBackbone", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class DeiTModel(ViTModel):
    """DeiT trunk returning the final feature as a 4D map."""

    KMODELS_CONFIG = DEIT_CONFIG
    KMODELS_WEIGHTS = DEIT_WEIGHTS
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

    def __init__(self, name="DeiTModel", **kwargs):
        super().__init__(name=name, **kwargs)
