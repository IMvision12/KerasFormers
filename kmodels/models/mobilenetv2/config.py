"""MobileNetV2 variant registry (timm-ported).

Variant ids follow timm naming: ``<arch>_<recipe>_<train_dataset>``.
``timm_id`` is the corresponding timm repo on HuggingFace Hub, used
by both the offline conversion script and the runtime ``timm:`` path
on :meth:`MobileNetV2.from_timm`.
"""

_WM50 = {"width_multiplier": 0.5, "depth_multiplier": 1.0, "fix_channels": False}
_WM100 = {"width_multiplier": 1.0, "depth_multiplier": 1.0, "fix_channels": False}
_WM110 = {"width_multiplier": 1.1, "depth_multiplier": 1.2, "fix_channels": True}
_WM120 = {"width_multiplier": 1.2, "depth_multiplier": 1.4, "fix_channels": True}
_WM140 = {"width_multiplier": 1.4, "depth_multiplier": 1.0, "fix_channels": False}


def _v(arch, timm_id, image_size=224, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


MOBILENETV2_CONFIG = {
    "mobilenetv2_050_lamb_in1k": _v(_WM50, "mobilenetv2_050.lamb_in1k"),
    "mobilenetv2_100_ra_in1k": _v(_WM100, "mobilenetv2_100.ra_in1k"),
    "mobilenetv2_110d_ra_in1k": _v(_WM110, "mobilenetv2_110d.ra_in1k"),
    "mobilenetv2_120d_ra_in1k": _v(_WM120, "mobilenetv2_120d.ra_in1k"),
    "mobilenetv2_140_ra_in1k": _v(_WM140, "mobilenetv2_140.ra_in1k"),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
MOBILENETV2_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"}
    for variant in MOBILENETV2_CONFIG
}
