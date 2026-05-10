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

RT_DETR_V2_CONFIG = {
    "rtdetr-v2-r18vd": _r18,
    "rtdetr-v2-r34vd": _r34,
    "rtdetr-v2-r50vd": _r50,
    "rtdetr-v2-r101vd": _r101,
}

RT_DETR_V2_WEIGHTS = {
    "rtdetr-v2-r18vd": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/rt-detrv2/rtdetr_v2_r18vd_coco.weights.h5",
    },
    "rtdetr-v2-r34vd": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/rt-detrv2/rtdetr_v2_r34vd_coco.weights.h5",
    },
    "rtdetr-v2-r50vd": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/rt-detrv2/rtdetr_v2_r50vd_coco.weights.h5",
    },
    "rtdetr-v2-r101vd": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/rt-detrv2/rtdetr_v2_r101vd_coco.weights.h5",
    },
}
