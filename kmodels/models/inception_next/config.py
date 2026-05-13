"""InceptionNeXt variant registry (timm-ported)."""

_ATTO = {
    "depths": [2, 2, 6, 2],
    "num_filters": [40, 80, 160, 320],
    "mlp_ratios": [4, 4, 4, 3],
    "band_kernel_size": 9,
    "branch_ratio": 0.25,
}
_TINY = {
    "depths": [3, 3, 9, 3],
    "num_filters": [96, 192, 384, 768],
    "mlp_ratios": [4, 4, 4, 3],
    "band_kernel_size": 11,
    "branch_ratio": 0.125,
}
_SMALL = {
    "depths": [3, 3, 27, 3],
    "num_filters": [96, 192, 384, 768],
    "mlp_ratios": [4, 4, 4, 3],
    "band_kernel_size": 11,
    "branch_ratio": 0.125,
}
_BASE = {
    "depths": [3, 3, 27, 3],
    "num_filters": [128, 256, 512, 1024],
    "mlp_ratios": [4, 4, 4, 3],
    "band_kernel_size": 11,
    "branch_ratio": 0.125,
}


def _v(arch, timm_id, image_size=224, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


INCEPTION_NEXT_CONFIG = {
    "inception_next_atto_sail_in1k": _v(_ATTO, "inception_next_atto.sail_in1k"),
    "inception_next_tiny_sail_in1k": _v(_TINY, "inception_next_tiny.sail_in1k"),
    "inception_next_small_sail_in1k": _v(_SMALL, "inception_next_small.sail_in1k"),
    "inception_next_base_sail_in1k": _v(_BASE, "inception_next_base.sail_in1k"),
    "inception_next_base_sail_in1k_384": _v(
        _BASE, "inception_next_base.sail_in1k_384", image_size=384
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
INCEPTION_NEXT_WEIGHTS = {
    "inception_next_atto_sail_in1k": {
        "url": f"{_BASE_URL}/inception_next_atto_sail_in1k.weights.h5"
    },
    "inception_next_tiny_sail_in1k": {
        "url": f"{_BASE_URL}/inception_next_tiny_sail_in1k.weights.h5"
    },
    "inception_next_small_sail_in1k": {
        "url": f"{_BASE_URL}/inception_next_small_sail_in1k.weights.h5"
    },
    "inception_next_base_sail_in1k": {
        "url": f"{_BASE_URL}/inception_next_base_sail_in1k.weights.h5"
    },
    "inception_next_base_sail_in1k_384": {
        "url": f"{_BASE_URL}/inception_next_base_sail_in1k_384.weights.h5"
    },
}
