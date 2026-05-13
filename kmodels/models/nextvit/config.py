"""NextViT variant registry (timm-ported)."""

_NEXTVIT_SMALL = {
    "depths": [3, 4, 10, 3],
    "stem_chs": [64, 32, 64],
    "head_dim": 32,
    "mix_block_ratio": 0.75,
    "sr_ratios": [8, 4, 2, 1],
    "drop_path_rate": 0.1,
}
_NEXTVIT_BASE = {
    "depths": [3, 4, 20, 3],
    "stem_chs": [64, 32, 64],
    "head_dim": 32,
    "mix_block_ratio": 0.75,
    "sr_ratios": [8, 4, 2, 1],
    "drop_path_rate": 0.1,
}
_NEXTVIT_LARGE = {
    "depths": [3, 4, 30, 3],
    "stem_chs": [64, 32, 64],
    "head_dim": 32,
    "mix_block_ratio": 0.75,
    "sr_ratios": [8, 4, 2, 1],
    "drop_path_rate": 0.1,
}


def _v(arch, timm_id, image_size=224, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


NEXTVIT_CONFIG = {
    "nextvit_small_bd_in1k": _v(_NEXTVIT_SMALL, "nextvit_small.bd_in1k"),
    "nextvit_small_bd_in1k_384": _v(
        _NEXTVIT_SMALL, "nextvit_small.bd_in1k_384", image_size=384
    ),
    "nextvit_small_bd_ssld_6m_in1k": _v(
        _NEXTVIT_SMALL, "nextvit_small.bd_ssld_6m_in1k"
    ),
    "nextvit_small_bd_ssld_6m_in1k_384": _v(
        _NEXTVIT_SMALL, "nextvit_small.bd_ssld_6m_in1k_384", image_size=384
    ),
    "nextvit_base_bd_in1k": _v(_NEXTVIT_BASE, "nextvit_base.bd_in1k"),
    "nextvit_base_bd_in1k_384": _v(
        _NEXTVIT_BASE, "nextvit_base.bd_in1k_384", image_size=384
    ),
    "nextvit_base_bd_ssld_6m_in1k": _v(_NEXTVIT_BASE, "nextvit_base.bd_ssld_6m_in1k"),
    "nextvit_base_bd_ssld_6m_in1k_384": _v(
        _NEXTVIT_BASE, "nextvit_base.bd_ssld_6m_in1k_384", image_size=384
    ),
    "nextvit_large_bd_in1k": _v(_NEXTVIT_LARGE, "nextvit_large.bd_in1k"),
    "nextvit_large_bd_in1k_384": _v(
        _NEXTVIT_LARGE, "nextvit_large.bd_in1k_384", image_size=384
    ),
    "nextvit_large_bd_ssld_6m_in1k": _v(
        _NEXTVIT_LARGE, "nextvit_large.bd_ssld_6m_in1k"
    ),
    "nextvit_large_bd_ssld_6m_in1k_384": _v(
        _NEXTVIT_LARGE, "nextvit_large.bd_ssld_6m_in1k_384", image_size=384
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/nextvit"
NEXTVIT_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in NEXTVIT_CONFIG
}
