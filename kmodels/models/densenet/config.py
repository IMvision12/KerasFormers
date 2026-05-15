DENSENET_MODEL_CONFIG = {
    "densenet121_tv_in1k": {
        "num_blocks": [6, 12, 24, 16],
        "growth_rate": 32,
        "initial_filter": 64,
        "timm_id": "densenet121.tv_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "densenet161_tv_in1k": {
        "num_blocks": [6, 12, 36, 24],
        "growth_rate": 48,
        "initial_filter": 96,
        "timm_id": "densenet161.tv_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "densenet169_tv_in1k": {
        "num_blocks": [6, 12, 32, 32],
        "growth_rate": 32,
        "initial_filter": 64,
        "timm_id": "densenet169.tv_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "densenet201_tv_in1k": {
        "num_blocks": [6, 12, 48, 32],
        "growth_rate": 32,
        "initial_filter": 64,
        "timm_id": "densenet201.tv_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "densenet264d_ra_in1k": {
        "num_blocks": [6, 12, 64, 48],
        "growth_rate": 48,
        "initial_filter": 96,
        "timm_id": "densenet264d.ra_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
}

DENSENET_WEIGHT_CONFIG = {
    "densenet121_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.1/densenet121_tv_in1k.weights.h5",
    },
    "densenet161_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.1/densenet161_tv_in1k.weights.h5",
    },
    "densenet169_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.1/densenet169_tv_in1k.weights.h5",
    },
    "densenet201_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.1/densenet201_tv_in1k.weights.h5",
    },
    "densenet264d_ra_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.1/densenet264d_ra_in1k.weights.h5",
    },
}
