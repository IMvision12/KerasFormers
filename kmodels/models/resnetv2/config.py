"""ResNetV2 / BiT variant registry (timm-ported)."""


def _v(arch, timm_id, image_size, num_classes):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


_50X1 = {"block_repeats": [3, 4, 6, 3], "width_factor": 1}
_50X3 = {"block_repeats": [3, 4, 6, 3], "width_factor": 3}
_101X1 = {"block_repeats": [3, 4, 23, 3], "width_factor": 1}
_101X3 = {"block_repeats": [3, 4, 23, 3], "width_factor": 3}
_152X2 = {"block_repeats": [3, 8, 36, 3], "width_factor": 2}
_152X4 = {"block_repeats": [3, 8, 36, 3], "width_factor": 4}

_IN21K = 21843
_IN1K = 1000


RESNETV2_CONFIG = {
    "resnetv2_50x1_bit_goog_in21k": _v(
        _50X1, "resnetv2_50x1_bit.goog_in21k", 224, _IN21K
    ),
    "resnetv2_50x1_bit_goog_in21k_ft_in1k": _v(
        _50X1, "resnetv2_50x1_bit.goog_in21k_ft_in1k", 448, _IN1K
    ),
    "resnetv2_50x3_bit_goog_in21k": _v(
        _50X3, "resnetv2_50x3_bit.goog_in21k", 224, _IN21K
    ),
    "resnetv2_50x3_bit_goog_in21k_ft_in1k": _v(
        _50X3, "resnetv2_50x3_bit.goog_in21k_ft_in1k", 448, _IN1K
    ),
    "resnetv2_101x1_bit_goog_in21k": _v(
        _101X1, "resnetv2_101x1_bit.goog_in21k", 224, _IN21K
    ),
    "resnetv2_101x1_bit_goog_in21k_ft_in1k": _v(
        _101X1, "resnetv2_101x1_bit.goog_in21k_ft_in1k", 448, _IN1K
    ),
    "resnetv2_101x3_bit_goog_in21k": _v(
        _101X3, "resnetv2_101x3_bit.goog_in21k", 224, _IN21K
    ),
    "resnetv2_101x3_bit_goog_in21k_ft_in1k": _v(
        _101X3, "resnetv2_101x3_bit.goog_in21k_ft_in1k", 448, _IN1K
    ),
    "resnetv2_152x2_bit_goog_in21k": _v(
        _152X2, "resnetv2_152x2_bit.goog_in21k", 224, _IN21K
    ),
    "resnetv2_152x2_bit_goog_in21k_ft_in1k": _v(
        _152X2, "resnetv2_152x2_bit.goog_in21k_ft_in1k", 448, _IN1K
    ),
    "resnetv2_152x4_bit_goog_in21k": _v(
        _152X4, "resnetv2_152x4_bit.goog_in21k", 224, _IN21K
    ),
    "resnetv2_152x4_bit_goog_in21k_ft_in1k": _v(
        _152X4, "resnetv2_152x4_bit.goog_in21k_ft_in1k", 480, _IN1K
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
RESNETV2_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in RESNETV2_CONFIG
}
RESNETV2_WEIGHTS["resnetv2_152x4_bit_goog_in21k"] = {
    "url": f"{_BASE_URL}/resnetv2_152x4_bit_goog_in21k.weights.json"
}
RESNETV2_WEIGHTS["resnetv2_152x4_bit_goog_in21k_ft_in1k"] = {
    "url": f"{_BASE_URL}/resnetv2_152x4_bit_goog_in21k_ft_in1k.weights.json"
}
