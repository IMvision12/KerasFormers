MOBILENETV3_MODEL_CONFIG = {
    "mobilenetv3_small_050_lamb_in1k": {
        "width_multiplier": 0.5,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": False,
        "timm_id": "mobilenetv3_small_050.lamb_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv3_small_075_lamb_in1k": {
        "width_multiplier": 0.75,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": False,
        "timm_id": "mobilenetv3_small_075.lamb_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv3_small_100_lamb_in1k": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": False,
        "timm_id": "mobilenetv3_small_100.lamb_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv3_large_075_ra_in1k": {
        "width_multiplier": 0.75,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": False,
        "timm_id": "mobilenetv3_large_075.ra_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv3_large_100_ra_in1k": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": False,
        "timm_id": "mobilenetv3_large_100.ra_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv3_small_minimal_100_in1k": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": True,
        "timm_id": "tf_mobilenetv3_small_minimal_100.in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv3_large_minimal_100_in1k": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": True,
        "timm_id": "tf_mobilenetv3_large_minimal_100.in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
}

MOBILENETV3_WEIGHT_CONFIG = {
    "mobilenetv3_small_050_lamb_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/mobilenetv3_small_050_lamb_in1k.weights.h5",
    },
    "mobilenetv3_small_075_lamb_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/mobilenetv3_small_075_lamb_in1k.weights.h5",
    },
    "mobilenetv3_small_100_lamb_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/mobilenetv3_small_100_lamb_in1k.weights.h5",
    },
    "mobilenetv3_large_075_ra_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/mobilenetv3_large_075_ra_in1k.weights.h5",
    },
    "mobilenetv3_large_100_ra_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/mobilenetv3_large_100_ra_in1k.weights.h5",
    },
    "mobilenetv3_small_minimal_100_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/mobilenetv3_small_minimal_100_in1k.weights.h5",
    },
    "mobilenetv3_large_minimal_100_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/mobilenetv3_large_minimal_100_in1k.weights.h5",
    },
}
