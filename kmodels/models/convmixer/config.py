"""ConvMixer variant registry (timm-ported)."""

_CONVMIXER_1536_20 = {
    "dim": 1536,
    "depth": 20,
    "patch_size": 7,
    "kernel_size": 9,
    "activation": "gelu",
}
_CONVMIXER_768_32 = {
    "dim": 768,
    "depth": 32,
    "patch_size": 7,
    "kernel_size": 7,
    "activation": "relu",
}
_CONVMIXER_1024_20_KS9_P14 = {
    "dim": 1024,
    "depth": 20,
    "patch_size": 14,
    "kernel_size": 9,
    "activation": "gelu",
}


def _v(arch, timm_id, image_size=224, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


CONVMIXER_CONFIG = {
    "convmixer_1536_20_in1k": _v(_CONVMIXER_1536_20, "convmixer_1536_20.in1k"),
    "convmixer_768_32_in1k": _v(_CONVMIXER_768_32, "convmixer_768_32.in1k"),
    "convmixer_1024_20_ks9_p14_in1k": _v(
        _CONVMIXER_1024_20_KS9_P14, "convmixer_1024_20_ks9_p14.in1k"
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
CONVMIXER_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"}
    for variant in CONVMIXER_CONFIG
}
