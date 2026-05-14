"""ConvNeXtV2 as thin :class:`ConvNeXt` subclasses (timm-ported)."""

import keras

from kmodels.models.convnext.convert_convnext_torch_to_keras import (
    transfer_convnext_weights,
)
from kmodels.models.convnext.convnext_model import (
    ConvNeXtBackbone,
    ConvNeXtClassify,
    ConvNeXtModel,
)
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import CONVNEXTV2_CONFIG, CONVNEXTV2_WEIGHTS


@keras.saving.register_keras_serializable(package="kmodels")
class ConvNeXtV2Classify(ConvNeXtClassify):
    """ConvNeXtV2 classifier (GRN + post-FCMAE finetune).

    Reference:
    - [ConvNeXt V2](https://arxiv.org/abs/2301.00808)

    Construction:

    >>> ConvNeXtV2Classify.from_weights("convnextv2_base_fcmae_ft_in22k_in1k")
    >>> ConvNeXtV2Classify.from_weights("timm:timm/convnextv2_base.fcmae_ft_in22k_in1k")
    """

    KMODELS_CONFIG = CONVNEXTV2_CONFIG
    KMODELS_WEIGHTS = CONVNEXTV2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_convnext_weights(keras_model, state_dict)

    def __init__(self, name="ConvNeXtV2Classify", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class ConvNeXtV2Backbone(ConvNeXtBackbone):
    """ConvNeXtV2 feature extractor."""

    KMODELS_CONFIG = CONVNEXTV2_CONFIG
    KMODELS_WEIGHTS = CONVNEXTV2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = ConvNeXtV2Classify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_convnext_weights(keras_model, state_dict)

    def __init__(self, name="ConvNeXtV2Backbone", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class ConvNeXtV2Model(ConvNeXtModel):
    """ConvNeXtV2 trunk returning the final stage feature map ``(B, H, W, C)``."""

    KMODELS_CONFIG = CONVNEXTV2_CONFIG
    KMODELS_WEIGHTS = CONVNEXTV2_WEIGHTS
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = ConvNeXtV2Classify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_convnext_weights(keras_model, state_dict)

    def __init__(self, name="ConvNeXtV2Model", **kwargs):
        super().__init__(name=name, **kwargs)
