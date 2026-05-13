"""DeiT / DeiT3 variant registry (timm-ported)."""

_DEIT_TINY = {"patch_size": 16, "dim": 192, "depth": 12, "num_heads": 3}
_DEIT_SMALL = {"patch_size": 16, "dim": 384, "depth": 12, "num_heads": 6}
_DEIT_BASE = {"patch_size": 16, "dim": 768, "depth": 12, "num_heads": 12}

_DEIT_TINY_DIST = {**_DEIT_TINY, "use_distillation": True}
_DEIT_SMALL_DIST = {**_DEIT_SMALL, "use_distillation": True}
_DEIT_BASE_DIST = {**_DEIT_BASE, "use_distillation": True}

_DEIT3_SMALL = {**_DEIT_SMALL, "no_embed_class": True, "init_values": 1e-6}
_DEIT3_MEDIUM = {
    "patch_size": 16,
    "dim": 512,
    "depth": 12,
    "num_heads": 8,
    "no_embed_class": True,
    "init_values": 1e-6,
}
_DEIT3_BASE = {**_DEIT_BASE, "no_embed_class": True, "init_values": 1e-6}
_DEIT3_LARGE = {
    "patch_size": 16,
    "dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "no_embed_class": True,
    "init_values": 1e-6,
}
_DEIT3_HUGE = {
    "patch_size": 14,
    "dim": 1280,
    "depth": 32,
    "num_heads": 16,
    "no_embed_class": True,
    "init_values": 1e-6,
}

_IN1K = 1000


def _v(arch, timm_id, image_size, num_classes=_IN1K):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


DEIT_CONFIG = {
    "deit_tiny_patch16_224_fb_in1k": _v(
        _DEIT_TINY, "deit_tiny_patch16_224.fb_in1k", 224
    ),
    "deit_small_patch16_224_fb_in1k": _v(
        _DEIT_SMALL, "deit_small_patch16_224.fb_in1k", 224
    ),
    "deit_base_patch16_224_fb_in1k": _v(
        _DEIT_BASE, "deit_base_patch16_224.fb_in1k", 224
    ),
    "deit_base_patch16_384_fb_in1k": _v(
        _DEIT_BASE, "deit_base_patch16_384.fb_in1k", 384
    ),
    "deit_tiny_distilled_patch16_224_fb_in1k": _v(
        _DEIT_TINY_DIST, "deit_tiny_distilled_patch16_224.fb_in1k", 224
    ),
    "deit_small_distilled_patch16_224_fb_in1k": _v(
        _DEIT_SMALL_DIST, "deit_small_distilled_patch16_224.fb_in1k", 224
    ),
    "deit_base_distilled_patch16_224_fb_in1k": _v(
        _DEIT_BASE_DIST, "deit_base_distilled_patch16_224.fb_in1k", 224
    ),
    "deit_base_distilled_patch16_384_fb_in1k": _v(
        _DEIT_BASE_DIST, "deit_base_distilled_patch16_384.fb_in1k", 384
    ),
    "deit3_small_patch16_224_fb_in1k": _v(
        _DEIT3_SMALL, "deit3_small_patch16_224.fb_in1k", 224
    ),
    "deit3_small_patch16_384_fb_in1k": _v(
        _DEIT3_SMALL, "deit3_small_patch16_384.fb_in1k", 384
    ),
    "deit3_small_patch16_224_fb_in22k_ft_in1k": _v(
        _DEIT3_SMALL, "deit3_small_patch16_224.fb_in22k_ft_in1k", 224
    ),
    "deit3_small_patch16_384_fb_in22k_ft_in1k": _v(
        _DEIT3_SMALL, "deit3_small_patch16_384.fb_in22k_ft_in1k", 384
    ),
    "deit3_medium_patch16_224_fb_in1k": _v(
        _DEIT3_MEDIUM, "deit3_medium_patch16_224.fb_in1k", 224
    ),
    "deit3_medium_patch16_224_fb_in22k_ft_in1k": _v(
        _DEIT3_MEDIUM, "deit3_medium_patch16_224.fb_in22k_ft_in1k", 224
    ),
    "deit3_base_patch16_224_fb_in1k": _v(
        _DEIT3_BASE, "deit3_base_patch16_224.fb_in1k", 224
    ),
    "deit3_base_patch16_384_fb_in1k": _v(
        _DEIT3_BASE, "deit3_base_patch16_384.fb_in1k", 384
    ),
    "deit3_base_patch16_224_fb_in22k_ft_in1k": _v(
        _DEIT3_BASE, "deit3_base_patch16_224.fb_in22k_ft_in1k", 224
    ),
    "deit3_base_patch16_384_fb_in22k_ft_in1k": _v(
        _DEIT3_BASE, "deit3_base_patch16_384.fb_in22k_ft_in1k", 384
    ),
    "deit3_large_patch16_224_fb_in1k": _v(
        _DEIT3_LARGE, "deit3_large_patch16_224.fb_in1k", 224
    ),
    "deit3_large_patch16_384_fb_in1k": _v(
        _DEIT3_LARGE, "deit3_large_patch16_384.fb_in1k", 384
    ),
    "deit3_large_patch16_224_fb_in22k_ft_in1k": _v(
        _DEIT3_LARGE, "deit3_large_patch16_224.fb_in22k_ft_in1k", 224
    ),
    "deit3_large_patch16_384_fb_in22k_ft_in1k": _v(
        _DEIT3_LARGE, "deit3_large_patch16_384.fb_in22k_ft_in1k", 384
    ),
    "deit3_huge_patch14_224_fb_in1k": _v(
        _DEIT3_HUGE, "deit3_huge_patch14_224.fb_in1k", 224
    ),
    "deit3_huge_patch14_224_fb_in22k_ft_in1k": _v(
        _DEIT3_HUGE, "deit3_huge_patch14_224.fb_in22k_ft_in1k", 224
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
DEIT_WEIGHTS = {
    variant: {
        "url": (
            f"{_BASE_URL}/{variant}.weights.json"
            if variant.startswith("deit3_huge")
            else f"{_BASE_URL}/{variant}.weights.h5"
        )
    }
    for variant in DEIT_CONFIG
}
