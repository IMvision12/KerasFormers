DENSENET_MODEL_CONFIG = {
    "densenet121": {
        "depths": [6, 12, 24, 16],
        "growth_rate": 32,
        "initial_filter": 64,
        "image_size": 224,
        "num_classes": 1000,
    },
    "densenet161": {
        "depths": [6, 12, 36, 24],
        "growth_rate": 48,
        "initial_filter": 96,
        "image_size": 224,
        "num_classes": 1000,
    },
    "densenet169": {
        "depths": [6, 12, 32, 32],
        "growth_rate": 32,
        "initial_filter": 64,
        "image_size": 224,
        "num_classes": 1000,
    },
    "densenet201": {
        "depths": [6, 12, 48, 32],
        "growth_rate": 32,
        "initial_filter": 64,
        "image_size": 224,
        "num_classes": 1000,
    },
}

DENSENET_WEIGHTS_URLS = {
    "densenet121_tv_in1k": {
        "model": "densenet121",
        "timm_id": "densenet121.tv_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/densenet121_tv_in1k.weights.h5",
    },
    "densenet161_tv_in1k": {
        "model": "densenet161",
        "timm_id": "densenet161.tv_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/densenet161_tv_in1k.weights.h5",
    },
    "densenet169_tv_in1k": {
        "model": "densenet169",
        "timm_id": "densenet169.tv_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/densenet169_tv_in1k.weights.h5",
    },
    "densenet201_tv_in1k": {
        "model": "densenet201",
        "timm_id": "densenet201.tv_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/densenet201_tv_in1k.weights.h5",
    },
}
