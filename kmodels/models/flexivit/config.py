"""FlexiViT variant registry (timm-ported)."""

_SMALL = {
    "patch_size": 16,
    "dim": 384,
    "depth": 12,
    "num_heads": 6,
    "no_embed_class": True,
}
_BASE = {
    "patch_size": 16,
    "dim": 768,
    "depth": 12,
    "num_heads": 12,
    "no_embed_class": True,
}
_LARGE = {
    "patch_size": 16,
    "dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "no_embed_class": True,
}

_IN1K = 1000
_IN21K = 21843


def _v(arch, timm_id, image_size=240, num_classes=_IN1K):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


FLEXIVIT_CONFIG = {
    "flexivit_small_1200ep_in1k": _v(_SMALL, "flexivit_small.1200ep_in1k"),
    "flexivit_small_600ep_in1k": _v(_SMALL, "flexivit_small.600ep_in1k"),
    "flexivit_small_300ep_in1k": _v(_SMALL, "flexivit_small.300ep_in1k"),
    "flexivit_base_1200ep_in1k": _v(_BASE, "flexivit_base.1200ep_in1k"),
    "flexivit_base_300ep_in1k": _v(_BASE, "flexivit_base.300ep_in1k"),
    "flexivit_base_1000ep_in21k": _v(
        _BASE, "flexivit_base.1000ep_in21k", num_classes=_IN21K
    ),
    "flexivit_base_300ep_in21k": _v(
        _BASE, "flexivit_base.300ep_in21k", num_classes=_IN21K
    ),
    "flexivit_large_1200ep_in1k": _v(_LARGE, "flexivit_large.1200ep_in1k"),
    "flexivit_large_600ep_in1k": _v(_LARGE, "flexivit_large.600ep_in1k"),
    "flexivit_large_300ep_in1k": _v(_LARGE, "flexivit_large.300ep_in1k"),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
FLEXIVIT_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in FLEXIVIT_CONFIG
}
