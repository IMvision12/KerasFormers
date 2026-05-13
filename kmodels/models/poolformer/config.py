"""PoolFormer variant registry (timm-ported)."""

_POOLFORMER_S12 = {
    "embed_dims": (64, 128, 320, 512),
    "num_blocks": (2, 2, 6, 2),
    "init_scale": 1e-5,
}
_POOLFORMER_S24 = {
    "embed_dims": (64, 128, 320, 512),
    "num_blocks": (4, 4, 12, 4),
    "init_scale": 1e-5,
}
_POOLFORMER_S36 = {
    "embed_dims": (64, 128, 320, 512),
    "num_blocks": (6, 6, 18, 6),
    "init_scale": 1e-6,
}
_POOLFORMER_M36 = {
    "embed_dims": (96, 192, 384, 768),
    "num_blocks": (6, 6, 18, 6),
    "init_scale": 1e-6,
}
_POOLFORMER_M48 = {
    "embed_dims": (96, 192, 384, 768),
    "num_blocks": (8, 8, 24, 8),
    "init_scale": 1e-6,
}


def _v(arch, timm_id, image_size=224, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


POOLFORMER_CONFIG = {
    "poolformer_s12_sail_in1k": _v(_POOLFORMER_S12, "poolformer_s12.sail_in1k"),
    "poolformer_s24_sail_in1k": _v(_POOLFORMER_S24, "poolformer_s24.sail_in1k"),
    "poolformer_s36_sail_in1k": _v(_POOLFORMER_S36, "poolformer_s36.sail_in1k"),
    "poolformer_m36_sail_in1k": _v(_POOLFORMER_M36, "poolformer_m36.sail_in1k"),
    "poolformer_m48_sail_in1k": _v(_POOLFORMER_M48, "poolformer_m48.sail_in1k"),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
POOLFORMER_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"}
    for variant in POOLFORMER_CONFIG
}
