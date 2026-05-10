DETR_CONFIG = {
    "detr-resnet-50": {
        "backbone_variant": "ResNet50",
    },
    "detr-resnet-101": {
        "backbone_variant": "ResNet101",
    },
}

DETR_WEIGHTS = {
    "detr-resnet-50": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/DeTR/detr_resnet_50_coco.weights.h5",
    },
    "detr-resnet-101": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/DeTR/detr_resnet_101_coco.weights.h5",
    },
}
