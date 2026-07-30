DEEPLABV3_CONFIG = {
    "deeplabv3_resnet50_coco_voc": {
        "backbone_variant": "ResNet50",
        "num_classes": 21,
        "image_size": 520,
    },
    "deeplabv3_resnet101_coco_voc": {
        "backbone_variant": "ResNet101",
        "num_classes": 21,
        "image_size": 520,
    },
}

DEEPLABV3_WEIGHTS_URLS = {
    "deeplabv3_resnet50_coco_voc": {
        "url": "https://huggingface.co/kerasformers/deeplabv3_resnet50_coco_voc",
    },
    "deeplabv3_resnet101_coco_voc": {
        "url": "https://huggingface.co/kerasformers/deeplabv3_resnet101_coco_voc",
    },
}
