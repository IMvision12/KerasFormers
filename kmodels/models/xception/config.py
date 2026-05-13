"""Xception variant registry.

This kmodels Xception implements the original Keras / Chollet 2017 Xception
architecture (entry/middle/exit flow). The single release variant warm-starts
from the keras.applications.Xception checkpoint that was converted via
``convert_xception_org_keras_to_keras.py``.

Note: timm's aligned Xception family (``xception41``, ``xception65``,
``xception71``, ``xception41p``, ``xception65p``) uses a different,
groups-aware Aligned Xception backbone and is not implemented here.
"""

_XCEPTION = {}  # Fixed architecture; no arch kwargs.


def _v(arch, timm_id, image_size=299, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


XCEPTION_CONFIG = {
    # timm doesn't host the original-Keras Xception weights; we use the
    # legacy keras-applications port. ``timm_id`` is set to the closest
    # canonical name to keep the registry uniform.
    "xception_in1k": _v(_XCEPTION, "xception.tf_in1k"),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
XCEPTION_WEIGHTS = {
    "xception_in1k": {"url": f"{_BASE_URL}/keras_org_xception.weights.h5"},
}
