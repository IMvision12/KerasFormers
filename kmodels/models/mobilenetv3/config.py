"""MobileNetV3 variant registry (timm-ported).

Variant ids follow timm naming: ``<arch>_<recipe>_<train_dataset>``.
``timm_id`` is the corresponding timm repo on HuggingFace Hub, used
by both the offline conversion script and the runtime ``timm:`` path
on :meth:`MobileNetV3.from_timm`.
"""

_SMALL_050 = {
    "width_multiplier": 0.5,
    "depth_multiplier": 1.0,
    "config": "small",
    "minimal": False,
}
_SMALL_075 = {
    "width_multiplier": 0.75,
    "depth_multiplier": 1.0,
    "config": "small",
    "minimal": False,
}
_SMALL_100 = {
    "width_multiplier": 1.0,
    "depth_multiplier": 1.0,
    "config": "small",
    "minimal": False,
}
_LARGE_075 = {
    "width_multiplier": 0.75,
    "depth_multiplier": 1.0,
    "config": "large",
    "minimal": False,
}
_LARGE_100 = {
    "width_multiplier": 1.0,
    "depth_multiplier": 1.0,
    "config": "large",
    "minimal": False,
}
_SMALL_MIN_100 = {
    "width_multiplier": 1.0,
    "depth_multiplier": 1.0,
    "config": "small",
    "minimal": True,
}
_LARGE_MIN_100 = {
    "width_multiplier": 1.0,
    "depth_multiplier": 1.0,
    "config": "large",
    "minimal": True,
}


def _v(arch, timm_id, image_size=224, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


MOBILENETV3_CONFIG = {
    "mobilenetv3_small_050_lamb_in1k": _v(
        _SMALL_050, "mobilenetv3_small_050.lamb_in1k"
    ),
    "mobilenetv3_small_075_lamb_in1k": _v(
        _SMALL_075, "mobilenetv3_small_075.lamb_in1k"
    ),
    "mobilenetv3_small_100_lamb_in1k": _v(
        _SMALL_100, "mobilenetv3_small_100.lamb_in1k"
    ),
    "mobilenetv3_large_075_ra_in1k": _v(_LARGE_075, "mobilenetv3_large_075.ra_in1k"),
    "mobilenetv3_large_100_ra_in1k": _v(_LARGE_100, "mobilenetv3_large_100.ra_in1k"),
    "mobilenetv3_small_minimal_100_in1k": _v(
        _SMALL_MIN_100, "tf_mobilenetv3_small_minimal_100.in1k"
    ),
    "mobilenetv3_large_minimal_100_in1k": _v(
        _LARGE_MIN_100, "tf_mobilenetv3_large_minimal_100.in1k"
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
MOBILENETV3_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"}
    for variant in MOBILENETV3_CONFIG
}
