"""Swin Transformer variant registry (timm-ported)."""

_TINY_P4W7 = {
    "window_size": 7,
    "embed_dim": 96,
    "depths": (2, 2, 6, 2),
    "num_heads": (3, 6, 12, 24),
    "pretrain_size": 224,
}
_SMALL_P4W7 = {
    "window_size": 7,
    "embed_dim": 96,
    "depths": (2, 2, 18, 2),
    "num_heads": (3, 6, 12, 24),
    "pretrain_size": 224,
}
_BASE_P4W7 = {
    "window_size": 7,
    "embed_dim": 128,
    "depths": (2, 2, 18, 2),
    "num_heads": (4, 8, 16, 32),
    "pretrain_size": 224,
}
_BASE_P4W12 = {
    "window_size": 12,
    "embed_dim": 128,
    "depths": (2, 2, 18, 2),
    "num_heads": (4, 8, 16, 32),
    "pretrain_size": 384,
}
_LARGE_P4W7 = {
    "window_size": 7,
    "embed_dim": 192,
    "depths": (2, 2, 18, 2),
    "num_heads": (6, 12, 24, 48),
    "pretrain_size": 224,
}
_LARGE_P4W12 = {
    "window_size": 12,
    "embed_dim": 192,
    "depths": (2, 2, 18, 2),
    "num_heads": (6, 12, 24, 48),
    "pretrain_size": 384,
}

_IN1K = 1000
_IN22K = 21841


def _v(arch, timm_id, num_classes):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": arch["pretrain_size"],
        "num_classes": num_classes,
    }


SWIN_CONFIG = {
    "swin_tiny_patch4_window7_224_ms_in1k": _v(
        _TINY_P4W7, "swin_tiny_patch4_window7_224.ms_in1k", _IN1K
    ),
    "swin_tiny_patch4_window7_224_ms_in22k": _v(
        _TINY_P4W7, "swin_tiny_patch4_window7_224.ms_in22k", _IN22K
    ),
    "swin_small_patch4_window7_224_ms_in1k": _v(
        _SMALL_P4W7, "swin_small_patch4_window7_224.ms_in1k", _IN1K
    ),
    "swin_small_patch4_window7_224_ms_in22k": _v(
        _SMALL_P4W7, "swin_small_patch4_window7_224.ms_in22k", _IN22K
    ),
    "swin_small_patch4_window7_224_ms_in22k_ft_in1k": _v(
        _SMALL_P4W7, "swin_small_patch4_window7_224.ms_in22k_ft_in1k", _IN1K
    ),
    "swin_base_patch4_window7_224_ms_in1k": _v(
        _BASE_P4W7, "swin_base_patch4_window7_224.ms_in1k", _IN1K
    ),
    "swin_base_patch4_window7_224_ms_in22k": _v(
        _BASE_P4W7, "swin_base_patch4_window7_224.ms_in22k", _IN22K
    ),
    "swin_base_patch4_window7_224_ms_in22k_ft_in1k": _v(
        _BASE_P4W7, "swin_base_patch4_window7_224.ms_in22k_ft_in1k", _IN1K
    ),
    "swin_base_patch4_window12_384_ms_in1k": _v(
        _BASE_P4W12, "swin_base_patch4_window12_384.ms_in1k", _IN1K
    ),
    "swin_base_patch4_window12_384_ms_in22k": _v(
        _BASE_P4W12, "swin_base_patch4_window12_384.ms_in22k", _IN22K
    ),
    "swin_base_patch4_window12_384_ms_in22k_ft_in1k": _v(
        _BASE_P4W12, "swin_base_patch4_window12_384.ms_in22k_ft_in1k", _IN1K
    ),
    "swin_large_patch4_window7_224_ms_in22k": _v(
        _LARGE_P4W7, "swin_large_patch4_window7_224.ms_in22k", _IN22K
    ),
    "swin_large_patch4_window7_224_ms_in22k_ft_in1k": _v(
        _LARGE_P4W7, "swin_large_patch4_window7_224.ms_in22k_ft_in1k", _IN1K
    ),
    "swin_large_patch4_window12_384_ms_in22k": _v(
        _LARGE_P4W12, "swin_large_patch4_window12_384.ms_in22k", _IN22K
    ),
    "swin_large_patch4_window12_384_ms_in22k_ft_in1k": _v(
        _LARGE_P4W12, "swin_large_patch4_window12_384.ms_in22k_ft_in1k", _IN1K
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/swin"
SWIN_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in SWIN_CONFIG
}
