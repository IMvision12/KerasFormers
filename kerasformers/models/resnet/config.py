RESNET_MODEL_CONFIG = {
    "resnet50": {
        "depths": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
    },
    "resnet101": {
        "depths": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
    },
    "resnet152": {
        "depths": [3, 8, 36, 3],
        "filters": [64, 128, 256, 512],
    },
}

RESNET_WEIGHT_CONFIG = {
    "resnet50_tv_in1k": {
        "model": "resnet50",
        "timm_id": "resnet50.tv_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/resnet50_tv_in1k.weights.h5",
    },
    "resnet50_a1_in1k": {
        "model": "resnet50",
        "timm_id": "resnet50.a1_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/resnet50_a1_in1k.weights.h5",
    },
    "resnet50_gluon_in1k": {
        "model": "resnet50",
        "timm_id": "resnet50.gluon_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/resnet50_gluon_in1k.weights.h5",
    },
    "resnet101_tv_in1k": {
        "model": "resnet101",
        "timm_id": "resnet101.tv_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/resnet101_tv_in1k.weights.h5",
    },
    "resnet101_a1_in1k": {
        "model": "resnet101",
        "timm_id": "resnet101.a1_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/resnet101_a1_in1k.weights.h5",
    },
    "resnet101_gluon_in1k": {
        "model": "resnet101",
        "timm_id": "resnet101.gluon_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/resnet101_gluon_in1k.weights.h5",
    },
    "resnet152_tv_in1k": {
        "model": "resnet152",
        "timm_id": "resnet152.tv_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/resnet152_tv_in1k.weights.h5",
    },
    "resnet152_a1_in1k": {
        "model": "resnet152",
        "timm_id": "resnet152.a1_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/resnet152_a1_in1k.weights.h5",
    },
    "resnet152_gluon_in1k": {
        "model": "resnet152",
        "timm_id": "resnet152.gluon_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/resnet152_gluon_in1k.weights.h5",
    },
}
