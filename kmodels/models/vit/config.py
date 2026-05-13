"""ViT variant registry (timm-ported).

Variant ids follow timm: ``vit_<size>_patch<N>_<resolution>_<recipe>_<dataset>``.
"""

_TINY16 = {
    "patch_size": 16,
    "dim": 192,
    "depth": 12,
    "num_heads": 3,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qk_norm": False,
}
_SMALL16 = {
    "patch_size": 16,
    "dim": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qk_norm": False,
}
_SMALL32 = {
    "patch_size": 32,
    "dim": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qk_norm": False,
}
_BASE16 = {
    "patch_size": 16,
    "dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qk_norm": False,
}
_BASE32 = {
    "patch_size": 32,
    "dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qk_norm": False,
}
_LARGE16 = {
    "patch_size": 16,
    "dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qk_norm": False,
}
_LARGE32 = {
    "patch_size": 32,
    "dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qk_norm": False,
}

_IN1K = 1000
_IN21K = 21843


def _v(arch, timm_id, image_size, num_classes):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


VIT_CONFIG = {
    "vit_tiny_patch16_224_augreg_in21k_ft_in1k": _v(
        _TINY16, "vit_tiny_patch16_224.augreg_in21k_ft_in1k", 224, _IN1K
    ),
    "vit_tiny_patch16_384_augreg_in21k_ft_in1k": _v(
        _TINY16, "vit_tiny_patch16_384.augreg_in21k_ft_in1k", 384, _IN1K
    ),
    "vit_tiny_patch16_224_augreg_in21k": _v(
        _TINY16, "vit_tiny_patch16_224.augreg_in21k", 224, _IN21K
    ),
    "vit_small_patch16_224_augreg_in21k_ft_in1k": _v(
        _SMALL16, "vit_small_patch16_224.augreg_in21k_ft_in1k", 224, _IN1K
    ),
    "vit_small_patch16_384_augreg_in21k_ft_in1k": _v(
        _SMALL16, "vit_small_patch16_384.augreg_in21k_ft_in1k", 384, _IN1K
    ),
    "vit_small_patch16_224_augreg_in1k": _v(
        _SMALL16, "vit_small_patch16_224.augreg_in1k", 224, _IN1K
    ),
    "vit_small_patch16_384_augreg_in1k": _v(
        _SMALL16, "vit_small_patch16_384.augreg_in1k", 384, _IN1K
    ),
    "vit_small_patch16_224_augreg_in21k": _v(
        _SMALL16, "vit_small_patch16_224.augreg_in21k", 224, _IN21K
    ),
    "vit_small_patch32_224_augreg_in21k_ft_in1k": _v(
        _SMALL32, "vit_small_patch32_224.augreg_in21k_ft_in1k", 224, _IN1K
    ),
    "vit_small_patch32_384_augreg_in21k_ft_in1k": _v(
        _SMALL32, "vit_small_patch32_384.augreg_in21k_ft_in1k", 384, _IN1K
    ),
    "vit_small_patch32_224_augreg_in21k": _v(
        _SMALL32, "vit_small_patch32_224.augreg_in21k", 224, _IN21K
    ),
    "vit_base_patch16_224_augreg_in21k_ft_in1k": _v(
        _BASE16, "vit_base_patch16_224.augreg_in21k_ft_in1k", 224, _IN1K
    ),
    "vit_base_patch16_384_augreg_in21k_ft_in1k": _v(
        _BASE16, "vit_base_patch16_384.augreg_in21k_ft_in1k", 384, _IN1K
    ),
    "vit_base_patch16_224_orig_in21k_ft_in1k": _v(
        _BASE16, "vit_base_patch16_224.orig_in21k_ft_in1k", 224, _IN1K
    ),
    "vit_base_patch16_384_orig_in21k_ft_in1k": _v(
        _BASE16, "vit_base_patch16_384.orig_in21k_ft_in1k", 384, _IN1K
    ),
    "vit_base_patch16_224_augreg_in1k": _v(
        _BASE16, "vit_base_patch16_224.augreg_in1k", 224, _IN1K
    ),
    "vit_base_patch16_384_augreg_in1k": _v(
        _BASE16, "vit_base_patch16_384.augreg_in1k", 384, _IN1K
    ),
    "vit_base_patch16_224_augreg_in21k": _v(
        _BASE16, "vit_base_patch16_224.augreg_in21k", 224, _IN21K
    ),
    "vit_base_patch32_224_augreg_in21k_ft_in1k": _v(
        _BASE32, "vit_base_patch32_224.augreg_in21k_ft_in1k", 224, _IN1K
    ),
    "vit_base_patch32_384_augreg_in21k_ft_in1k": _v(
        _BASE32, "vit_base_patch32_384.augreg_in21k_ft_in1k", 384, _IN1K
    ),
    "vit_base_patch32_224_augreg_in1k": _v(
        _BASE32, "vit_base_patch32_224.augreg_in1k", 224, _IN1K
    ),
    "vit_base_patch32_384_augreg_in1k": _v(
        _BASE32, "vit_base_patch32_384.augreg_in1k", 384, _IN1K
    ),
    "vit_base_patch32_224_augreg_in21k": _v(
        _BASE32, "vit_base_patch32_224.augreg_in21k", 224, _IN21K
    ),
    "vit_large_patch16_224_augreg_in21k_ft_in1k": _v(
        _LARGE16, "vit_large_patch16_224.augreg_in21k_ft_in1k", 224, _IN1K
    ),
    "vit_large_patch16_384_augreg_in21k_ft_in1k": _v(
        _LARGE16, "vit_large_patch16_384.augreg_in21k_ft_in1k", 384, _IN1K
    ),
    "vit_large_patch16_224_augreg_in21k": _v(
        _LARGE16, "vit_large_patch16_224.augreg_in21k", 224, _IN21K
    ),
    "vit_large_patch32_384_orig_in21k_ft_in1k": _v(
        _LARGE32, "vit_large_patch32_384.orig_in21k_ft_in1k", 384, _IN1K
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/vit"
VIT_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in VIT_CONFIG
}
