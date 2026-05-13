"""Res2Net variant registry (timm-ported)."""


def _v(arch, timm_id):
    return {**arch, "timm_id": timm_id}


RES2NET_CONFIG = {
    "res2net50_26w_4s_in1k": _v(
        {"depth": [3, 4, 6, 3], "base_width": 26, "scale": 4, "cardinality": 1},
        "res2net50_26w_4s.in1k",
    ),
    "res2net101_26w_4s_in1k": _v(
        {"depth": [3, 4, 23, 3], "base_width": 26, "scale": 4, "cardinality": 1},
        "res2net101_26w_4s.in1k",
    ),
    "res2net50_26w_6s_in1k": _v(
        {"depth": [3, 4, 6, 3], "base_width": 26, "scale": 6, "cardinality": 1},
        "res2net50_26w_6s.in1k",
    ),
    "res2net50_26w_8s_in1k": _v(
        {"depth": [3, 4, 6, 3], "base_width": 26, "scale": 8, "cardinality": 1},
        "res2net50_26w_8s.in1k",
    ),
    "res2net50_48w_2s_in1k": _v(
        {"depth": [3, 4, 6, 3], "base_width": 48, "scale": 2, "cardinality": 1},
        "res2net50_48w_2s.in1k",
    ),
    "res2net50_14w_8s_in1k": _v(
        {"depth": [3, 4, 6, 3], "base_width": 14, "scale": 8, "cardinality": 1},
        "res2net50_14w_8s.in1k",
    ),
    "res2next50_in1k": _v(
        {"depth": [3, 4, 6, 3], "base_width": 4, "scale": 4, "cardinality": 8},
        "res2next50.in1k",
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
RES2NET_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in RES2NET_CONFIG
}
