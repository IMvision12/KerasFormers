"""MaxViT variant registry (timm-ported)."""

_TINY = {
    "stem_width": 64,
    "depths": [2, 2, 5, 2],
    "embed_dim": [64, 128, 256, 512],
    "num_heads": [2, 4, 8, 16],
}
_SMALL = {
    "stem_width": 64,
    "depths": [2, 2, 5, 2],
    "embed_dim": [96, 192, 384, 768],
    "num_heads": [3, 6, 12, 24],
}
_BASE = {
    "stem_width": 64,
    "depths": [2, 6, 14, 2],
    "embed_dim": [96, 192, 384, 768],
    "num_heads": [3, 6, 12, 24],
}
_LARGE = {
    "stem_width": 128,
    "depths": [2, 6, 14, 2],
    "embed_dim": [128, 256, 512, 1024],
    "num_heads": [4, 8, 16, 32],
}
_XLARGE = {
    "stem_width": 192,
    "depths": [2, 6, 14, 2],
    "embed_dim": [192, 384, 768, 1536],
    "num_heads": [6, 12, 24, 48],
}

_IN1K = 1000
_IN21K = 21843
_WS = {224: 7, 384: 12, 512: 16}


def _v(arch, timm_id, image_size, num_classes=_IN1K):
    return {
        **arch,
        "window_size": _WS[image_size],
        "image_size": image_size,
        "timm_id": timm_id,
        "num_classes": num_classes,
    }


MAXVIT_CONFIG = {
    "maxvit_tiny_tf_224_in1k": _v(_TINY, "maxvit_tiny_tf_224.in1k", 224),
    "maxvit_tiny_tf_384_in1k": _v(_TINY, "maxvit_tiny_tf_384.in1k", 384),
    "maxvit_tiny_tf_512_in1k": _v(_TINY, "maxvit_tiny_tf_512.in1k", 512),
    "maxvit_small_tf_224_in1k": _v(_SMALL, "maxvit_small_tf_224.in1k", 224),
    "maxvit_small_tf_384_in1k": _v(_SMALL, "maxvit_small_tf_384.in1k", 384),
    "maxvit_small_tf_512_in1k": _v(_SMALL, "maxvit_small_tf_512.in1k", 512),
    "maxvit_base_tf_224_in1k": _v(_BASE, "maxvit_base_tf_224.in1k", 224),
    "maxvit_base_tf_384_in1k": _v(_BASE, "maxvit_base_tf_384.in1k", 384),
    "maxvit_base_tf_512_in1k": _v(_BASE, "maxvit_base_tf_512.in1k", 512),
    "maxvit_base_tf_224_in21k": _v(
        _BASE, "maxvit_base_tf_224.in21k", 224, num_classes=_IN21K
    ),
    "maxvit_base_tf_384_in21k_ft_in1k": _v(
        _BASE, "maxvit_base_tf_384.in21k_ft_in1k", 384
    ),
    "maxvit_base_tf_512_in21k_ft_in1k": _v(
        _BASE, "maxvit_base_tf_512.in21k_ft_in1k", 512
    ),
    "maxvit_large_tf_224_in1k": _v(_LARGE, "maxvit_large_tf_224.in1k", 224),
    "maxvit_large_tf_384_in1k": _v(_LARGE, "maxvit_large_tf_384.in1k", 384),
    "maxvit_large_tf_512_in1k": _v(_LARGE, "maxvit_large_tf_512.in1k", 512),
    "maxvit_large_tf_224_in21k": _v(
        _LARGE, "maxvit_large_tf_224.in21k", 224, num_classes=_IN21K
    ),
    "maxvit_large_tf_384_in21k_ft_in1k": _v(
        _LARGE, "maxvit_large_tf_384.in21k_ft_in1k", 384
    ),
    "maxvit_large_tf_512_in21k_ft_in1k": _v(
        _LARGE, "maxvit_large_tf_512.in21k_ft_in1k", 512
    ),
    "maxvit_xlarge_tf_224_in21k": _v(
        _XLARGE, "maxvit_xlarge_tf_224.in21k", 224, num_classes=_IN21K
    ),
    "maxvit_xlarge_tf_384_in21k_ft_in1k": _v(
        _XLARGE, "maxvit_xlarge_tf_384.in21k_ft_in1k", 384
    ),
    "maxvit_xlarge_tf_512_in21k_ft_in1k": _v(
        _XLARGE, "maxvit_xlarge_tf_512.in21k_ft_in1k", 512
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/MaxViT"
MAXVIT_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in MAXVIT_CONFIG
}
