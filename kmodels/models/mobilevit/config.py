"""MobileViT variant registry (timm-ported).

Variant ids follow timm naming: ``<arch>_<recipe>_<train_dataset>``.
``timm_id`` is the corresponding timm repo on HuggingFace Hub, used
by both the offline conversion script and the runtime ``timm:`` path
on :meth:`MobileViT.from_timm`.
"""

_XXS = {
    "initial_dims": 16,
    "head_dims": 320,
    "block_dims": [16, 24, 48, 64, 80],
    "expansion_ratio": [2.0, 2.0, 2.0, 2.0, 2.0],
    "attention_dims": [None, None, 64, 80, 96],
}
_XS = {
    "initial_dims": 16,
    "head_dims": 384,
    "block_dims": [32, 48, 64, 80, 96],
    "expansion_ratio": [4.0, 4.0, 4.0, 4.0, 4.0],
    "attention_dims": [None, None, 96, 120, 144],
}
_S = {
    "initial_dims": 16,
    "head_dims": 640,
    "block_dims": [32, 64, 96, 128, 160],
    "expansion_ratio": [4.0, 4.0, 4.0, 4.0, 4.0],
    "attention_dims": [None, None, 144, 192, 240],
}


def _v(arch, timm_id, image_size=256, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


MOBILEVIT_CONFIG = {
    "mobilevit_xxs_cvnets_in1k": _v(_XXS, "mobilevit_xxs.cvnets_in1k"),
    "mobilevit_xs_cvnets_in1k": _v(_XS, "mobilevit_xs.cvnets_in1k"),
    "mobilevit_s_cvnets_in1k": _v(_S, "mobilevit_s.cvnets_in1k"),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
MOBILEVIT_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"}
    for variant in MOBILEVIT_CONFIG
}
