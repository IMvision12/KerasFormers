_r18 = {
    "backbone_hidden_sizes": (64, 128, 256, 512),
    "backbone_block_repeats": (2, 2, 2, 2),
    "backbone_layer_type": "basic",
    "encoder_in_channels": (128, 256, 512),
    "hidden_expansion": 0.5,
    "decoder_layers": 3,
}

_r34 = {
    "backbone_hidden_sizes": (64, 128, 256, 512),
    "backbone_block_repeats": (3, 4, 6, 3),
    "backbone_layer_type": "basic",
    "encoder_in_channels": (128, 256, 512),
    "hidden_expansion": 0.5,
    "decoder_layers": 4,
}

_r50 = {
    "backbone_hidden_sizes": (256, 512, 1024, 2048),
    "backbone_block_repeats": (3, 4, 6, 3),
    "backbone_layer_type": "bottleneck",
    "encoder_in_channels": (512, 1024, 2048),
    "decoder_layers": 6,
}

_r101 = {
    "backbone_hidden_sizes": (256, 512, 1024, 2048),
    "backbone_block_repeats": (3, 4, 23, 3),
    "backbone_layer_type": "bottleneck",
    "encoder_in_channels": (512, 1024, 2048),
    "encoder_hidden_dim": 384,
    "encoder_ffn_dim": 2048,
    "decoder_layers": 6,
}

RT_DETR_CONFIG = {
    "rtdetr-r18vd": _r18,
    "rtdetr-r18vd-coco-o365": _r18,
    "rtdetr-r34vd": _r34,
    "rtdetr-r50vd": _r50,
    "rtdetr-r50vd-coco-o365": _r50,
    "rtdetr-r101vd": _r101,
    "rtdetr-r101vd-coco-o365": _r101,
}

RT_DETR_WEIGHTS = {
    "rtdetr-r18vd": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rt-detr/rt_detr_r18vd.weights.h5",
    },
    "rtdetr-r18vd-coco-o365": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rt-detr/rt_detr_r18vd_coco_o365.weights.h5",
    },
    "rtdetr-r34vd": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rt-detr/rt_detr_r34vd.weights.h5",
    },
    "rtdetr-r50vd": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rt-detr/rt_detr_r50vd.weights.h5",
    },
    "rtdetr-r50vd-coco-o365": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rt-detr/rt_detr_r50vd_coco_o365.weights.h5",
    },
    "rtdetr-r101vd": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rt-detr/rt_detr_r101vd.weights.h5",
    },
    "rtdetr-r101vd-coco-o365": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rt-detr/rt_detr_r101vd_coco_o365.weights.h5",
    },
}
