"""EfficientNet-Lite variant registry (timm-ported)."""

DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
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


def _v(width, depth, dropout, image_size, timm_id, num_classes=1000):
    return {
        "width_coefficient": width,
        "depth_coefficient": depth,
        "dropout_rate": dropout,
        "default_size": image_size,
        "image_size": image_size,
        "timm_id": timm_id,
        "num_classes": num_classes,
    }


EFFICIENTNET_LITE_CONFIG = {
    "tf_efficientnet_lite0_in1k": _v(1.0, 1.0, 0.2, 224, "tf_efficientnet_lite0.in1k"),
    "tf_efficientnet_lite1_in1k": _v(1.0, 1.1, 0.2, 240, "tf_efficientnet_lite1.in1k"),
    "tf_efficientnet_lite2_in1k": _v(1.1, 1.2, 0.3, 260, "tf_efficientnet_lite2.in1k"),
    "tf_efficientnet_lite3_in1k": _v(1.2, 1.4, 0.3, 300, "tf_efficientnet_lite3.in1k"),
    "tf_efficientnet_lite4_in1k": _v(1.4, 1.8, 0.3, 380, "tf_efficientnet_lite4.in1k"),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
EFFICIENTNET_LITE_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"}
    for variant in EFFICIENTNET_LITE_CONFIG
}
