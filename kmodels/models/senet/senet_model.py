import keras

from kmodels.models.resnet.resnet_model import (
    ResNetBackbone,
    ResNetClassify,
    ResNetModel,
    bottleneck_block,
)
from kmodels.models.resnext.resnext_model import resnext_block

from .config import SENET_CONFIG, SENET_WEIGHTS

_BLOCK_FN_LOOKUP = {
    "bottleneck_block": bottleneck_block,
    "resnext_block": resnext_block,
}


def _resolve_block_fn(kwargs):
    """Convert a ``block_fn_name`` string (from the variant config) to a
    callable. Subclass __init__s call this to support both SE-ResNet
    (``bottleneck_block``) and SE-ResNeXt (``resnext_block``) variants.
    """
    name = kwargs.pop("block_fn_name", None)
    if name is not None:
        kwargs["block_fn"] = _BLOCK_FN_LOOKUP[name]


@keras.saving.register_keras_serializable(package="kmodels")
class SENetClassify(ResNetClassify):
    """Squeeze-and-Excitation ResNet / ResNeXt classifier.

    Covers both ``seresnet*`` (bottleneck block) and ``seresnext*``
    (grouped block) variants — block_fn is selected per-variant via the
    ``block_fn_name`` key in :data:`SENET_CONFIG`.

    >>> SENetClassify.from_weights("seresnet50_a1_in1k")
    >>> SENetClassify.from_weights("seresnext50_32x4d_racm_in1k")
    >>> SENetClassify.from_weights("timm:timm/seresnet50.a1_in1k")
    """

    KMODELS_CONFIG = SENET_CONFIG
    KMODELS_WEIGHTS = SENET_WEIGHTS

    def __init__(self, senet=True, name="SENetClassify", **kwargs):
        _resolve_block_fn(kwargs)
        super().__init__(senet=senet, name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class SENetModel(ResNetModel):
    """SE-ResNet / SE-ResNeXt trunk returning the final stage feature map."""

    KMODELS_CONFIG = SENET_CONFIG
    KMODELS_WEIGHTS = SENET_WEIGHTS

    @classmethod
    def _release_warm_start_cls(cls):
        return SENetClassify

    def __init__(self, senet=True, name="SENetModel", **kwargs):
        _resolve_block_fn(kwargs)
        super().__init__(senet=senet, name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class SENetBackbone(ResNetBackbone):
    """SE-ResNet / SE-ResNeXt feature extractor (no classifier head)."""

    KMODELS_CONFIG = SENET_CONFIG
    KMODELS_WEIGHTS = SENET_WEIGHTS

    @classmethod
    def _release_warm_start_cls(cls):
        return SENetClassify

    def __init__(self, senet=True, name="SENetBackbone", **kwargs):
        _resolve_block_fn(kwargs)
        super().__init__(senet=senet, name=name, **kwargs)
