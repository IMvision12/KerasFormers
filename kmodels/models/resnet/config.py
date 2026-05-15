RESNET_MODEL_CONFIG = {
    "resnet50_tv_in1k": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "timm_id": "resnet50.tv_in1k",
    },
    "resnet50_a1_in1k": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "timm_id": "resnet50.a1_in1k",
    },
    "resnet50_gluon_in1k": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "timm_id": "resnet50.gluon_in1k",
    },
    "resnet101_tv_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "timm_id": "resnet101.tv_in1k",
    },
    "resnet101_a1_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "timm_id": "resnet101.a1_in1k",
    },
    "resnet101_gluon_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "timm_id": "resnet101.gluon_in1k",
    },
    "resnet152_tv_in1k": {
        "block_repeats": [3, 8, 36, 3],
        "filters": [64, 128, 256, 512],
        "timm_id": "resnet152.tv_in1k",
    },
    "resnet152_a1_in1k": {
        "block_repeats": [3, 8, 36, 3],
        "filters": [64, 128, 256, 512],
        "timm_id": "resnet152.a1_in1k",
    },
    "resnet152_gluon_in1k": {
        "block_repeats": [3, 8, 36, 3],
        "filters": [64, 128, 256, 512],
        "timm_id": "resnet152.gluon_in1k",
    },
}

RESNET_WEIGHT_CONFIG = {
    "resnet50_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet50_tv_in1k.weights.h5",
    },
    "resnet50_a1_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet50_a1_in1k.weights.h5",
    },
    "resnet50_gluon_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet50_gluon_in1k.weights.h5",
    },
    "resnet101_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet101_tv_in1k.weights.h5",
    },
    "resnet101_a1_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet101_a1_in1k.weights.h5",
    },
    "resnet101_gluon_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet101_gluon_in1k.weights.h5",
    },
    "resnet152_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet152_tv_in1k.weights.h5",
    },
    "resnet152_a1_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet152_a1_in1k.weights.h5",
    },
    "resnet152_gluon_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet152_gluon_in1k.weights.h5",
    },
}
