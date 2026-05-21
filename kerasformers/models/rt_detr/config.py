RT_DETR_MODEL_CONFIG = {
    "rtdetrv1_r18": {
        "backbone_hidden_sizes": (64, 128, 256, 512),
        "backbone_block_repeats": (2, 2, 2, 2),
        "backbone_layer_type": "basic",
        "encoder_in_channels": (128, 256, 512),
        "hidden_expansion": 0.5,
        "decoder_layers": 3,
    },
    "rtdetrv1_r34": {
        "backbone_hidden_sizes": (64, 128, 256, 512),
        "backbone_block_repeats": (3, 4, 6, 3),
        "backbone_layer_type": "basic",
        "encoder_in_channels": (128, 256, 512),
        "hidden_expansion": 0.5,
        "decoder_layers": 4,
    },
    "rtdetrv1_r50": {
        "backbone_hidden_sizes": (256, 512, 1024, 2048),
        "backbone_block_repeats": (3, 4, 6, 3),
        "backbone_layer_type": "bottleneck",
        "encoder_in_channels": (512, 1024, 2048),
        "decoder_layers": 6,
    },
    "rtdetrv1_r101": {
        "backbone_hidden_sizes": (256, 512, 1024, 2048),
        "backbone_block_repeats": (3, 4, 23, 3),
        "backbone_layer_type": "bottleneck",
        "encoder_in_channels": (512, 1024, 2048),
        "encoder_hidden_dim": 384,
        "encoder_ffn_dim": 2048,
        "decoder_layers": 6,
    },
}

RT_DETR_WEIGHT_CONFIG = {
    "rtdetr-r18vd": {
        "model": "rtdetrv1_r18",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RT-DETR/rt_detr_r18vd.weights.h5",
    },
    "rtdetr-r18vd-coco-o365": {
        "model": "rtdetrv1_r18",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RT-DETR/rt_detr_r18vd_coco_o365.weights.h5",
    },
    "rtdetr-r34vd": {
        "model": "rtdetrv1_r34",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RT-DETR/rt_detr_r34vd.weights.h5",
    },
    "rtdetr-r50vd": {
        "model": "rtdetrv1_r50",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RT-DETR/rt_detr_r50vd.weights.h5",
    },
    "rtdetr-r50vd-coco-o365": {
        "model": "rtdetrv1_r50",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RT-DETR/rt_detr_r50vd_coco_o365.weights.h5",
    },
    "rtdetr-r101vd": {
        "model": "rtdetrv1_r101",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RT-DETR/rt_detr_r101vd.weights.h5",
    },
    "rtdetr-r101vd-coco-o365": {
        "model": "rtdetrv1_r101",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RT-DETR/rt_detr_r101vd_coco_o365.weights.h5",
    },
}
