"""DenseNet variant registry (timm-ported)."""

_DENSENET121 = {"num_blocks": [6, 12, 24, 16], "growth_rate": 32, "initial_filter": 64}
_DENSENET161 = {"num_blocks": [6, 12, 36, 24], "growth_rate": 48, "initial_filter": 96}
_DENSENET169 = {"num_blocks": [6, 12, 32, 32], "growth_rate": 32, "initial_filter": 64}
_DENSENET201 = {"num_blocks": [6, 12, 48, 32], "growth_rate": 32, "initial_filter": 64}
_DENSENET264D = {"num_blocks": [6, 12, 64, 48], "growth_rate": 48, "initial_filter": 96}


def _v(arch, timm_id, image_size=224, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


DENSENET_CONFIG = {
    "densenet121_tv_in1k": _v(_DENSENET121, "densenet121.tv_in1k"),
    "densenet161_tv_in1k": _v(_DENSENET161, "densenet161.tv_in1k"),
    "densenet169_tv_in1k": _v(_DENSENET169, "densenet169.tv_in1k"),
    "densenet201_tv_in1k": _v(_DENSENET201, "densenet201.tv_in1k"),
    "densenet264d_ra_in1k": _v(_DENSENET264D, "densenet264d.ra_in1k"),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
DENSENET_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in DENSENET_CONFIG
}
