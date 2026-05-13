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
    def _release_warm_start_cls(cls):
        return ConvNeXtV2Classify

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
    def _release_warm_start_cls(cls):
        return ConvNeXtV2Classify

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_convnext_weights(keras_model, state_dict)

    def __init__(self, name="ConvNeXtV2Model", **kwargs):
        super().__init__(name=name, **kwargs)
