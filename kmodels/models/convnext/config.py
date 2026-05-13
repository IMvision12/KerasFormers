"""ConvNeXt variant registry (timm-ported)."""

_ATTO = {"depths": [2, 2, 6, 2], "projection_dims": [40, 80, 160, 320]}
_FEMTO = {"depths": [2, 2, 6, 2], "projection_dims": [48, 96, 192, 384]}
_PICO = {"depths": [2, 2, 6, 2], "projection_dims": [64, 128, 256, 512]}
_NANO = {"depths": [2, 2, 8, 2], "projection_dims": [80, 160, 320, 640]}
_TINY = {"depths": [3, 3, 9, 3], "projection_dims": [96, 192, 384, 768]}
_SMALL = {"depths": [3, 3, 27, 3], "projection_dims": [96, 192, 384, 768]}
_BASE = {"depths": [3, 3, 27, 3], "projection_dims": [128, 256, 512, 1024]}
_LARGE = {"depths": [3, 3, 27, 3], "projection_dims": [192, 384, 768, 1536]}
_XLARGE = {"depths": [3, 3, 27, 3], "projection_dims": [256, 512, 1024, 2048]}

_IN1K = 1000
_IN22K = 21841


def _v(arch, timm_id, image_size=224, num_classes=_IN1K):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


CONVNEXT_CONFIG = {
    "convnext_atto_d2_in1k": _v(_ATTO, "convnext_atto.d2_in1k"),
    "convnext_femto_d1_in1k": _v(_FEMTO, "convnext_femto.d1_in1k"),
    "convnext_pico_d1_in1k": _v(_PICO, "convnext_pico.d1_in1k"),
    "convnext_nano_d1h_in1k": _v(_NANO, "convnext_nano.d1h_in1k"),
    "convnext_nano_in12k_ft_in1k": _v(_NANO, "convnext_nano.in12k_ft_in1k"),
    "convnext_tiny_fb_in1k": _v(_TINY, "convnext_tiny.fb_in1k"),
    "convnext_tiny_fb_in22k": _v(_TINY, "convnext_tiny.fb_in22k", num_classes=_IN22K),
    "convnext_tiny_fb_in22k_ft_in1k": _v(_TINY, "convnext_tiny.fb_in22k_ft_in1k"),
    "convnext_tiny_fb_in22k_ft_in1k_384": _v(
        _TINY, "convnext_tiny.fb_in22k_ft_in1k_384", image_size=384
    ),
    "convnext_small_fb_in1k": _v(_SMALL, "convnext_small.fb_in1k"),
    "convnext_small_fb_in22k": _v(
        _SMALL, "convnext_small.fb_in22k", num_classes=_IN22K
    ),
    "convnext_small_fb_in22k_ft_in1k": _v(_SMALL, "convnext_small.fb_in22k_ft_in1k"),
    "convnext_small_fb_in22k_ft_in1k_384": _v(
        _SMALL, "convnext_small.fb_in22k_ft_in1k_384", image_size=384
    ),
    "convnext_base_fb_in1k": _v(_BASE, "convnext_base.fb_in1k"),
    "convnext_base_fb_in22k": _v(_BASE, "convnext_base.fb_in22k", num_classes=_IN22K),
    "convnext_base_fb_in22k_ft_in1k": _v(_BASE, "convnext_base.fb_in22k_ft_in1k"),
    "convnext_base_fb_in22k_ft_in1k_384": _v(
        _BASE, "convnext_base.fb_in22k_ft_in1k_384", image_size=384
    ),
    "convnext_large_fb_in1k": _v(_LARGE, "convnext_large.fb_in1k"),
    "convnext_large_fb_in22k": _v(
        _LARGE, "convnext_large.fb_in22k", num_classes=_IN22K
    ),
    "convnext_large_fb_in22k_ft_in1k": _v(_LARGE, "convnext_large.fb_in22k_ft_in1k"),
    "convnext_large_fb_in22k_ft_in1k_384": _v(
        _LARGE, "convnext_large.fb_in22k_ft_in1k_384", image_size=384
    ),
    "convnext_xlarge_fb_in22k": _v(
        _XLARGE, "convnext_xlarge.fb_in22k", num_classes=_IN22K
    ),
    "convnext_xlarge_fb_in22k_ft_in1k": _v(_XLARGE, "convnext_xlarge.fb_in22k_ft_in1k"),
    "convnext_xlarge_fb_in22k_ft_in1k_384": _v(
        _XLARGE, "convnext_xlarge.fb_in22k_ft_in1k_384", image_size=384
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/convnext"
CONVNEXT_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in CONVNEXT_CONFIG
}
