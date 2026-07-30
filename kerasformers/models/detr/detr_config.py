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

DETR_WEIGHTS_URLS = {
    "detr-resnet-50": {
        "url": "https://huggingface.co/kerasformers/detr-resnet-50",
    },
    "detr-resnet-101": {
        "url": "https://huggingface.co/kerasformers/detr-resnet-101",
    },
}

DETR_SEGMENT_WEIGHTS_URLS = {
    "detr-resnet-50-panoptic": {
        "url": "https://huggingface.co/kerasformers/detr-resnet-50-panoptic",
    },
    "detr-resnet-101-panoptic": {
        "url": "https://huggingface.co/kerasformers/detr-resnet-101-panoptic",
    },
}
