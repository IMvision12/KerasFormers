DEEPLABV3_CONFIG = {
    "deeplabv3_resnet50_coco_voc": {
        "backbone_variant": "ResNet50",
        "num_classes": 21,
        "input_shape": (520, 520, 3),
    },
    "deeplabv3_resnet101_coco_voc": {
        "backbone_variant": "ResNet101",
        "num_classes": 21,
        "input_shape": (520, 520, 3),
    },
}

DEEPLABV3_WEIGHTS = {
    "deeplabv3_resnet50_coco_voc": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/deeplabv3/deeplabv3_resnet50_coco_voc.weights.h5",
    },
    "deeplabv3_resnet101_coco_voc": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/deeplabv3/deeplabv3_resnet101_coco_voc.weights.h5",
    },
}
