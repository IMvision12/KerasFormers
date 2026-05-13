"""CaiT variant registry (timm-ported)."""

_XXS24 = {"patch_size": 16, "embed_dim": 192, "depth": 24, "num_heads": 4}
_XXS36 = {"patch_size": 16, "embed_dim": 192, "depth": 36, "num_heads": 4}
_XS24 = {"patch_size": 16, "embed_dim": 288, "depth": 24, "num_heads": 6}
_S24 = {"patch_size": 16, "embed_dim": 384, "depth": 24, "num_heads": 8}
_S36 = {"patch_size": 16, "embed_dim": 384, "depth": 36, "num_heads": 8}
_M36 = {"patch_size": 16, "embed_dim": 768, "depth": 36, "num_heads": 16}
_M48 = {"patch_size": 16, "embed_dim": 768, "depth": 48, "num_heads": 16}


def _v(arch, timm_id, image_size, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


CAIT_CONFIG = {
    "cait_xxs24_224_fb_dist_in1k": _v(_XXS24, "cait_xxs24_224.fb_dist_in1k", 224),
    "cait_xxs24_384_fb_dist_in1k": _v(_XXS24, "cait_xxs24_384.fb_dist_in1k", 384),
    "cait_xxs36_224_fb_dist_in1k": _v(_XXS36, "cait_xxs36_224.fb_dist_in1k", 224),
    "cait_xxs36_384_fb_dist_in1k": _v(_XXS36, "cait_xxs36_384.fb_dist_in1k", 384),
    "cait_xs24_384_fb_dist_in1k": _v(_XS24, "cait_xs24_384.fb_dist_in1k", 384),
    "cait_s24_224_fb_dist_in1k": _v(_S24, "cait_s24_224.fb_dist_in1k", 224),
    "cait_s24_384_fb_dist_in1k": _v(_S24, "cait_s24_384.fb_dist_in1k", 384),
    "cait_s36_384_fb_dist_in1k": _v(_S36, "cait_s36_384.fb_dist_in1k", 384),
    "cait_m36_384_fb_dist_in1k": _v(_M36, "cait_m36_384.fb_dist_in1k", 384),
    "cait_m48_448_fb_dist_in1k": _v(_M48, "cait_m48_448.fb_dist_in1k", 448),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
CAIT_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in CAIT_CONFIG
}
