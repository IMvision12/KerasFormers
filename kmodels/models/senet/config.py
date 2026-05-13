"""SENet variant registry (timm-ported, SE-ResNet + SE-ResNeXt)."""


def _v(arch, timm_id, *, resnext=False):
    out = {**arch, "timm_id": timm_id, "senet": True}
    if resnext:
        out["block_fn_name"] = "resnext_block"
    return out


_SE_R50 = {
    "block_repeats": [3, 4, 6, 3],
    "filters": [64, 128, 256, 512],
}
_SE_RX50_32X4D = {
    "block_repeats": [3, 4, 6, 3],
    "filters": [64, 128, 256, 512],
    "groups": 32,
    "width_factor": 2,
}
_SE_RX101_32X4D = {
    "block_repeats": [3, 4, 23, 3],
    "filters": [64, 128, 256, 512],
    "groups": 32,
    "width_factor": 2,
}
_SE_RX101_32X8D = {
    "block_repeats": [3, 4, 23, 3],
    "filters": [64, 128, 256, 512],
    "groups": 32,
    "width_factor": 4,
}


SENET_CONFIG = {
    "seresnet50_a1_in1k": _v(_SE_R50, "seresnet50.a1_in1k"),
    "seresnext50_32x4d_racm_in1k": _v(
        _SE_RX50_32X4D, "seresnext50_32x4d.racm_in1k", resnext=True
    ),
    "seresnext50_32x4d_gluon_in1k": _v(
        _SE_RX50_32X4D, "seresnext50_32x4d.gluon_in1k", resnext=True
    ),
    "seresnext101_32x4d_gluon_in1k": _v(
        _SE_RX101_32X4D, "seresnext101_32x4d.gluon_in1k", resnext=True
    ),
    "seresnext101_32x8d_ah_in1k": _v(
        _SE_RX101_32X8D, "seresnext101_32x8d.ah_in1k", resnext=True
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
SENET_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in SENET_CONFIG
}
