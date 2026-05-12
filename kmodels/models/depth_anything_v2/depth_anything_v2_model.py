import keras

from kmodels.models.depth_anything_v1.depth_anything_v1_model import (
    DepthAnythingV1DepthEstimation,
    DepthAnythingV1Model,
)

from .config import DEPTHANYTHINGV2_CONFIG, DEPTHANYTHINGV2_WEIGHTS


@keras.saving.register_keras_serializable(package="kmodels")
class DepthAnythingV2Model(DepthAnythingV1Model):
    """Depth Anything V2 backbone + DPT neck (no depth-prediction head).

    V2 reuses V1's architecture end-to-end — only training data and
    weights differ (synthetic data + larger-capacity teacher model).
    This class inherits from :class:`DepthAnythingV1Model` and just
    swaps in the V2 weights config.

    Reference:
        - `Depth Anything V2 <https://arxiv.org/abs/2406.09414>`_
    """

    KMODELS_CONFIG = DEPTHANYTHINGV2_CONFIG
    KMODELS_WEIGHTS = None
    HF_MODEL_TYPE = "depth_anything"

    def __init__(self, name="DepthAnythingV2Model", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class DepthAnythingV2DepthEstimation(DepthAnythingV1DepthEstimation):
    """Depth Anything V2 full monocular depth estimator.

    Includes both the relative-depth checkpoints
    (``depth_anything_v2_{small,base,large}``) and the metric fine-tunes
    (``depth_anything_v2_metric_{indoor,outdoor}_{small,base,large}``). The metric
    variants set ``depth_estimation_type="metric"`` and a non-trivial
    ``max_depth`` (``20.0`` for indoor, ``80.0`` for outdoor).

    Inherits the architecture from :class:`DepthAnythingV1DepthEstimation` and
    just swaps the weights config.

    Reference:
        - `Depth Anything V2 <https://arxiv.org/abs/2406.09414>`_
    """

    KMODELS_CONFIG = DEPTHANYTHINGV2_CONFIG
    KMODELS_WEIGHTS = DEPTHANYTHINGV2_WEIGHTS
    HF_MODEL_TYPE = "depth_anything"

    def __init__(self, name="DepthAnythingV2DepthEstimation", **kwargs):
        super().__init__(name=name, **kwargs)
