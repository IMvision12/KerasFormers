INCEPTION_NEXT_MODEL_CONFIG = {
    "inception_next_atto": {
        "depths": [2, 2, 6, 2],
        "num_filters": [40, 80, 160, 320],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 9,
        "branch_ratio": 0.25,
        "image_size": 224,
        "num_classes": 1000,
    },
    "inception_next_tiny": {
        "depths": [3, 3, 9, 3],
        "num_filters": [96, 192, 384, 768],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "image_size": 224,
        "num_classes": 1000,
    },
    "inception_next_small": {
        "depths": [3, 3, 27, 3],
        "num_filters": [96, 192, 384, 768],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "image_size": 224,
        "num_classes": 1000,
    },
    "inception_next_base": {
        "depths": [3, 3, 27, 3],
        "num_filters": [128, 256, 512, 1024],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "image_size": 224,
        "num_classes": 1000,
    },
    "inception_next_base_384": {
        "depths": [3, 3, 27, 3],
        "num_filters": [128, 256, 512, 1024],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "image_size": 384,
        "num_classes": 1000,
    },
}

INCEPTION_NEXT_WEIGHT_CONFIG = {
    "inception_next_atto_sail_in1k": {
        "model": "inception_next_atto",
        "timm_id": "inception_next_atto.sail_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/inception_next_atto_sail_in1k.weights.h5",
    },
    "inception_next_tiny_sail_in1k": {
        "model": "inception_next_tiny",
        "timm_id": "inception_next_tiny.sail_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/inception_next_tiny_sail_in1k.weights.h5",
    },
    "inception_next_small_sail_in1k": {
        "model": "inception_next_small",
        "timm_id": "inception_next_small.sail_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/inception_next_small_sail_in1k.weights.h5",
    },
    "inception_next_base_sail_in1k": {
        "model": "inception_next_base",
        "timm_id": "inception_next_base.sail_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/inception_next_base_sail_in1k.weights.h5",
    },
    "inception_next_base_sail_in1k_384": {
        "model": "inception_next_base_384",
        "timm_id": "inception_next_base.sail_in1k_384",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/inception_next_base_sail_in1k_384.weights.h5",
    },
}
