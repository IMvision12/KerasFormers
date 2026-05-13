"""PiT variant registry (timm-ported)."""

_XS = {
    "patch_size": 16,
    "stride": 8,
    "embed_dim": [96, 192, 384],
    "depth": [2, 6, 4],
    "heads": [2, 4, 8],
    "mlp_ratio": 4,
}
_TI = {
    "patch_size": 16,
    "stride": 8,
    "embed_dim": [64, 128, 256],
    "depth": [2, 6, 4],
    "heads": [2, 4, 8],
    "mlp_ratio": 4,
}
_S = {
    "patch_size": 16,
    "stride": 8,
    "embed_dim": [144, 288, 576],
    "depth": [2, 6, 4],
    "heads": [3, 6, 12],
    "mlp_ratio": 4,
}
_B = {
    "patch_size": 14,
    "stride": 7,
    "embed_dim": [256, 512, 1024],
    "depth": [3, 6, 4],
    "heads": [4, 8, 16],
    "mlp_ratio": 4,
}


def _v(arch, timm_id, image_size=224, distilled=False, num_classes=1000):
    return {
        **arch,
        "distilled": distilled,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


PIT_CONFIG = {
    "pit_xs_224_in1k": _v(_XS, "pit_xs_224.in1k"),
    "pit_xs_distilled_224_in1k": _v(_XS, "pit_xs_distilled_224.in1k", distilled=True),
    "pit_ti_224_in1k": _v(_TI, "pit_ti_224.in1k"),
    "pit_ti_distilled_224_in1k": _v(_TI, "pit_ti_distilled_224.in1k", distilled=True),
    "pit_s_224_in1k": _v(_S, "pit_s_224.in1k"),
    "pit_s_distilled_224_in1k": _v(_S, "pit_s_distilled_224.in1k", distilled=True),
    "pit_b_224_in1k": _v(_B, "pit_b_224.in1k"),
    "pit_b_distilled_224_in1k": _v(_B, "pit_b_distilled_224.in1k", distilled=True),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
PIT_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in PIT_CONFIG
}
