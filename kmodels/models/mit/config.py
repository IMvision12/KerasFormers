"""MiT (Mix Transformer / SegFormer encoder) variant registry.

Weights are HF checkpoints from ``nvidia/mit-b{0..5}`` (ImageNet-1k).
"""

_B0 = {"embed_dims": [32, 64, 160, 256], "depths": [2, 2, 2, 2]}
_B1_5 = {"embed_dims": [64, 128, 320, 512]}

_IN1K = 1000


def _v(arch, hf_id, depths=None, image_size=224, num_classes=_IN1K):
    cfg = {**arch, "hf_id": hf_id, "image_size": image_size, "num_classes": num_classes}
    if depths is not None:
        cfg["depths"] = depths
    return cfg


MIT_CONFIG = {
    "mit_b0_in1k": _v(_B0, "nvidia/mit-b0"),
    "mit_b1_in1k": _v(_B1_5, "nvidia/mit-b1", depths=[2, 2, 2, 2]),
    "mit_b2_in1k": _v(_B1_5, "nvidia/mit-b2", depths=[3, 4, 6, 3]),
    "mit_b3_in1k": _v(_B1_5, "nvidia/mit-b3", depths=[3, 4, 18, 3]),
    "mit_b4_in1k": _v(_B1_5, "nvidia/mit-b4", depths=[3, 8, 27, 3]),
    "mit_b5_in1k": _v(_B1_5, "nvidia/mit-b5", depths=[3, 6, 40, 3]),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.3"
_RELEASE_NAME = {
    "mit_b0_in1k": "MiT_B0",
    "mit_b1_in1k": "MiT_B1",
    "mit_b2_in1k": "MiT_B2",
    "mit_b3_in1k": "MiT_B3",
    "mit_b4_in1k": "MiT_B4",
    "mit_b5_in1k": "MiT_B5",
}
MIT_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{_RELEASE_NAME[variant]}.weights.h5"}
    for variant in MIT_CONFIG
}
