"""SwinV2 variant registry (timm-ported)."""

_TINY = {
    "embed_dim": 96,
    "depths": (2, 2, 6, 2),
    "num_heads": (3, 6, 12, 24),
}
_SMALL = {
    "embed_dim": 96,
    "depths": (2, 2, 18, 2),
    "num_heads": (3, 6, 12, 24),
}
_BASE = {
    "embed_dim": 128,
    "depths": (2, 2, 18, 2),
    "num_heads": (4, 8, 16, 32),
}
_LARGE = {
    "embed_dim": 192,
    "depths": (2, 2, 18, 2),
    "num_heads": (6, 12, 24, 48),
}

_IN1K = 1000
_IN22K = 21841


def _v(
    arch,
    timm_id,
    window_size,
    pretrain_size,
    image_size,
    num_classes,
    pretrained_window_size=0,
):
    return {
        **arch,
        "window_size": window_size,
        "pretrain_size": pretrain_size,
        "pretrained_window_size": pretrained_window_size,
        "image_size": image_size,
        "timm_id": timm_id,
        "num_classes": num_classes,
    }


SWINV2_CONFIG = {
    "swinv2_tiny_window8_256_ms_in1k": _v(
        _TINY, "swinv2_tiny_window8_256.ms_in1k", 8, 256, 256, _IN1K
    ),
    "swinv2_tiny_window16_256_ms_in1k": _v(
        _TINY, "swinv2_tiny_window16_256.ms_in1k", 16, 256, 256, _IN1K
    ),
    "swinv2_small_window8_256_ms_in1k": _v(
        _SMALL, "swinv2_small_window8_256.ms_in1k", 8, 256, 256, _IN1K
    ),
    "swinv2_small_window16_256_ms_in1k": _v(
        _SMALL, "swinv2_small_window16_256.ms_in1k", 16, 256, 256, _IN1K
    ),
    "swinv2_base_window8_256_ms_in1k": _v(
        _BASE, "swinv2_base_window8_256.ms_in1k", 8, 256, 256, _IN1K
    ),
    "swinv2_base_window12_192_ms_in22k": _v(
        _BASE, "swinv2_base_window12_192.ms_in22k", 12, 192, 192, _IN22K
    ),
    "swinv2_base_window12to16_192to256_ms_in22k_ft_in1k": _v(
        _BASE,
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        16,
        192,
        256,
        _IN1K,
        pretrained_window_size=12,
    ),
    "swinv2_base_window12to24_192to384_ms_in22k_ft_in1k": _v(
        _BASE,
        "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        24,
        192,
        384,
        _IN1K,
        pretrained_window_size=12,
    ),
    "swinv2_base_window16_256_ms_in1k": _v(
        _BASE, "swinv2_base_window16_256.ms_in1k", 16, 256, 256, _IN1K
    ),
    "swinv2_large_window12_192_ms_in22k": _v(
        _LARGE, "swinv2_large_window12_192.ms_in22k", 12, 192, 192, _IN22K
    ),
    "swinv2_large_window12to16_192to256_ms_in22k_ft_in1k": _v(
        _LARGE,
        "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
        16,
        192,
        256,
        _IN1K,
        pretrained_window_size=12,
    ),
    "swinv2_large_window12to24_192to384_ms_in22k_ft_in1k": _v(
        _LARGE,
        "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
        24,
        192,
        384,
        _IN1K,
        pretrained_window_size=12,
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/swin"
SWINV2_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in SWINV2_CONFIG
}
