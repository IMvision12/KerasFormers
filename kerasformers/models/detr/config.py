DETR_CONFIG = {
    "detr-resnet-50": {
        "backbone_variant": "ResNet50",
    },
    "detr-resnet-101": {
        "backbone_variant": "ResNet101",
    },
}

DETR_SEGMENT_CONFIG = {
    "detr-resnet-50-panoptic": {
        "backbone_variant": "ResNet50",
        "num_classes": 251,
    },
    "detr-resnet-101-panoptic": {
        "backbone_variant": "ResNet101",
        "num_classes": 251,
    },
}

DETR_WEIGHTS = {
    "detr-resnet-50": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/detr/detr_resnet50.weights.h5",
    },
    "detr-resnet-101": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/detr/detr_resnet101.weights.h5",
    },
}

DETR_SEGMENT_WEIGHTS = {
    "detr-resnet-50-panoptic": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/detr/detr_resnet50_panoptic.weights.h5",
    },
    "detr-resnet-101-panoptic": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/detr/detr_resnet101_panoptic.weights.h5",
    },
}
