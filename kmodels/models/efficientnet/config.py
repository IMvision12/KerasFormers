"""EfficientNet (TF) variant registry (timm-ported)."""

_B0 = {"width_coefficient": 1.0, "depth_coefficient": 1.0, "dropout_rate": 0.2}
_B1 = {"width_coefficient": 1.0, "depth_coefficient": 1.1, "dropout_rate": 0.2}
_B2 = {"width_coefficient": 1.1, "depth_coefficient": 1.2, "dropout_rate": 0.3}
_B3 = {"width_coefficient": 1.2, "depth_coefficient": 1.4, "dropout_rate": 0.3}
_B4 = {"width_coefficient": 1.4, "depth_coefficient": 1.8, "dropout_rate": 0.4}
_B5 = {"width_coefficient": 1.6, "depth_coefficient": 2.2, "dropout_rate": 0.4}
_B6 = {"width_coefficient": 1.8, "depth_coefficient": 2.6, "dropout_rate": 0.5}
_B7 = {"width_coefficient": 2.0, "depth_coefficient": 3.1, "dropout_rate": 0.5}
_B8 = {"width_coefficient": 2.2, "depth_coefficient": 3.6, "dropout_rate": 0.5}
_L2 = {"width_coefficient": 4.3, "depth_coefficient": 5.3, "dropout_rate": 0.5}

_DEFAULT_SIZE = {
    "b0": 224,
    "b1": 240,
    "b2": 260,
    "b3": 300,
    "b4": 380,
    "b5": 456,
    "b6": 528,
    "b7": 600,
    "b8": 672,
    "l2": 800,
}


DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal"},
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 1.0 / 3.0, "mode": "fan_out", "distribution": "uniform"},
}


def _v(arch, size, timm_id, image_size=None, num_classes=1000):
    return {
        **arch,
        "default_size": _DEFAULT_SIZE[size],
        "timm_id": timm_id,
        "image_size": image_size or _DEFAULT_SIZE[size],
        "num_classes": num_classes,
    }


_RECIPES = ["ns_jft_in1k", "ap_in1k", "aa_in1k", "in1k"]


def _b_variants(arch_dict, size, has_recipes):
    out = {}
    for recipe in has_recipes:
        variant = f"tf_efficientnet_{size}_{recipe}"
        timm_id = f"tf_efficientnet_{size}.{recipe}"
        out[variant] = _v(arch_dict, size, timm_id)
    return out


EFFICIENTNET_CONFIG = {
    **_b_variants(_B0, "b0", _RECIPES),
    **_b_variants(_B1, "b1", _RECIPES),
    **_b_variants(_B2, "b2", _RECIPES),
    **_b_variants(_B3, "b3", _RECIPES),
    **_b_variants(_B4, "b4", _RECIPES),
    **_b_variants(_B5, "b5", _RECIPES),
    **_b_variants(_B6, "b6", ["ns_jft_in1k", "ap_in1k", "aa_in1k"]),
    **_b_variants(_B7, "b7", ["ns_jft_in1k", "ap_in1k", "aa_in1k"]),
    **_b_variants(_B8, "b8", ["ap_in1k"]),
    "tf_efficientnet_l2_ns_jft_in1k": _v(_L2, "l2", "tf_efficientnet_l2.ns_jft_in1k"),
    "tf_efficientnet_l2_ns_jft_in1k_475": _v(
        _L2, "l2", "tf_efficientnet_l2.ns_jft_in1k_475", image_size=475
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
EFFICIENTNET_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"}
    for variant in EFFICIENTNET_CONFIG
}
