"""ResMLP variant registry (timm-ported)."""

_RESMLP12 = {
    "patch_size": 16,
    "embed_dim": 384,
    "depth": 12,
    "mlp_ratio": 4,
    "init_values": 1e-4,
}
_RESMLP24 = {
    "patch_size": 16,
    "embed_dim": 384,
    "depth": 24,
    "mlp_ratio": 4,
    "init_values": 1e-5,
}
_RESMLP36 = {
    "patch_size": 16,
    "embed_dim": 384,
    "depth": 36,
    "mlp_ratio": 4,
    "init_values": 1e-6,
}
_RESMLP_BIG_24 = {
    "patch_size": 8,
    "embed_dim": 768,
    "depth": 24,
    "mlp_ratio": 4,
    "init_values": 1e-6,
}


def _v(arch, timm_id, image_size=224, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


RESMLP_CONFIG = {
    "resmlp_12_224_fb_in1k": _v(_RESMLP12, "resmlp_12_224.fb_in1k"),
    "resmlp_12_224_fb_distilled_in1k": _v(_RESMLP12, "resmlp_12_224.fb_distilled_in1k"),
    "resmlp_24_224_fb_in1k": _v(_RESMLP24, "resmlp_24_224.fb_in1k"),
    "resmlp_24_224_fb_distilled_in1k": _v(_RESMLP24, "resmlp_24_224.fb_distilled_in1k"),
    "resmlp_36_224_fb_in1k": _v(_RESMLP36, "resmlp_36_224.fb_in1k"),
    "resmlp_36_224_fb_distilled_in1k": _v(_RESMLP36, "resmlp_36_224.fb_distilled_in1k"),
    "resmlp_big_24_224_fb_in1k": _v(_RESMLP_BIG_24, "resmlp_big_24_224.fb_in1k"),
    "resmlp_big_24_224_fb_distilled_in1k": _v(
        _RESMLP_BIG_24, "resmlp_big_24_224.fb_distilled_in1k"
    ),
    "resmlp_big_24_224_fb_in22k_ft_in1k": _v(
        _RESMLP_BIG_24, "resmlp_big_24_224.fb_in22k_ft_in1k"
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
RESMLP_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in RESMLP_CONFIG
}
