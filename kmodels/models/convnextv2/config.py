"""ConvNeXtV2 variant registry (timm-ported)."""

_ATTO = {
    "depths": [2, 2, 6, 2],
    "projection_dims": [40, 80, 160, 320],
    "use_conv": True,
}
_FEMTO = {
    "depths": [2, 2, 6, 2],
    "projection_dims": [48, 96, 192, 384],
    "use_conv": True,
}
_PICO = {
    "depths": [2, 2, 6, 2],
    "projection_dims": [64, 128, 256, 512],
    "use_conv": True,
}
_NANO = {
    "depths": [2, 2, 8, 2],
    "projection_dims": [80, 160, 320, 640],
    "use_conv": True,
}
_TINY = {"depths": [3, 3, 9, 3], "projection_dims": [96, 192, 384, 768]}
_BASE = {"depths": [3, 3, 27, 3], "projection_dims": [128, 256, 512, 1024]}
_LARGE = {"depths": [3, 3, 27, 3], "projection_dims": [192, 384, 768, 1536]}
_HUGE = {"depths": [3, 3, 27, 3], "projection_dims": [352, 704, 1408, 2816]}

_IN1K = 1000


def _v(arch, timm_id, image_size=224, num_classes=_IN1K):
    return {
        **arch,
        "use_grn": True,
        "layer_scale_init_value": None,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


CONVNEXTV2_CONFIG = {
    "convnextv2_atto_fcmae_ft_in1k": _v(_ATTO, "convnextv2_atto.fcmae_ft_in1k"),
    "convnextv2_femto_fcmae_ft_in1k": _v(_FEMTO, "convnextv2_femto.fcmae_ft_in1k"),
    "convnextv2_pico_fcmae_ft_in1k": _v(_PICO, "convnextv2_pico.fcmae_ft_in1k"),
    "convnextv2_nano_fcmae_ft_in1k": _v(_NANO, "convnextv2_nano.fcmae_ft_in1k"),
    "convnextv2_nano_fcmae_ft_in22k_in1k": _v(
        _NANO, "convnextv2_nano.fcmae_ft_in22k_in1k"
    ),
    "convnextv2_nano_fcmae_ft_in22k_in1k_384": _v(
        _NANO, "convnextv2_nano.fcmae_ft_in22k_in1k_384", image_size=384
    ),
    "convnextv2_tiny_fcmae_ft_in1k": _v(_TINY, "convnextv2_tiny.fcmae_ft_in1k"),
    "convnextv2_tiny_fcmae_ft_in22k_in1k": _v(
        _TINY, "convnextv2_tiny.fcmae_ft_in22k_in1k"
    ),
    "convnextv2_tiny_fcmae_ft_in22k_in1k_384": _v(
        _TINY, "convnextv2_tiny.fcmae_ft_in22k_in1k_384", image_size=384
    ),
    "convnextv2_base_fcmae_ft_in1k": _v(_BASE, "convnextv2_base.fcmae_ft_in1k"),
    "convnextv2_base_fcmae_ft_in22k_in1k": _v(
        _BASE, "convnextv2_base.fcmae_ft_in22k_in1k"
    ),
    "convnextv2_base_fcmae_ft_in22k_in1k_384": _v(
        _BASE, "convnextv2_base.fcmae_ft_in22k_in1k_384", image_size=384
    ),
    "convnextv2_large_fcmae_ft_in1k": _v(_LARGE, "convnextv2_large.fcmae_ft_in1k"),
    "convnextv2_large_fcmae_ft_in22k_in1k": _v(
        _LARGE, "convnextv2_large.fcmae_ft_in22k_in1k"
    ),
    "convnextv2_large_fcmae_ft_in22k_in1k_384": _v(
        _LARGE, "convnextv2_large.fcmae_ft_in22k_in1k_384", image_size=384
    ),
    "convnextv2_huge_fcmae_ft_in1k": _v(_HUGE, "convnextv2_huge.fcmae_ft_in1k"),
    "convnextv2_huge_fcmae_ft_in22k_in1k_384": _v(
        _HUGE, "convnextv2_huge.fcmae_ft_in22k_in1k_384", image_size=384
    ),
    "convnextv2_huge_fcmae_ft_in22k_in1k_512": _v(
        _HUGE, "convnextv2_huge.fcmae_ft_in22k_in1k_512", image_size=512
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/convnext"
CONVNEXTV2_WEIGHTS = {
    variant: {
        "url": (
            f"{_BASE_URL}/{variant}.weights.json"
            if variant.startswith("convnextv2_huge")
            else f"{_BASE_URL}/{variant}.weights.h5"
        )
    }
    for variant in CONVNEXTV2_CONFIG
}
