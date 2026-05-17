RT_DETR_V2_MODEL_CONFIG = {
    "rtdetrv2_r18": {
        "backbone_hidden_sizes": (64, 128, 256, 512),
        "backbone_block_repeats": (2, 2, 2, 2),
        "backbone_layer_type": "basic",
        "encoder_in_channels": (128, 256, 512),
        "hidden_expansion": 0.5,
        "decoder_layers": 3,
    },
    "rtdetrv2_r34": {
        "backbone_hidden_sizes": (64, 128, 256, 512),
        "backbone_block_repeats": (3, 4, 6, 3),
        "backbone_layer_type": "basic",
        "encoder_in_channels": (128, 256, 512),
        "hidden_expansion": 0.5,
        "decoder_layers": 4,
    },
    "rtdetrv2_r50": {
        "backbone_hidden_sizes": (256, 512, 1024, 2048),
        "backbone_block_repeats": (3, 4, 6, 3),
        "backbone_layer_type": "bottleneck",
        "encoder_in_channels": (512, 1024, 2048),
        "decoder_layers": 6,
    },
    "rtdetrv2_r101": {
        "backbone_hidden_sizes": (256, 512, 1024, 2048),
        "backbone_block_repeats": (3, 4, 23, 3),
        "backbone_layer_type": "bottleneck",
        "encoder_in_channels": (512, 1024, 2048),
        "encoder_hidden_dim": 384,
        "encoder_ffn_dim": 2048,
        "decoder_layers": 6,
    },
}

RT_DETR_V2_WEIGHT_CONFIG = {
    "rtdetr-v2-r18vd": {
        "model": "rtdetrv2_r18",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rt-detr-v2/rt_detr_v2_r18vd.weights.h5",
    },
    "rtdetr-v2-r34vd": {
        "model": "rtdetrv2_r34",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rt-detr-v2/rt_detr_v2_r34vd.weights.h5",
    },
    "rtdetr-v2-r50vd": {
        "model": "rtdetrv2_r50",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rt-detr-v2/rt_detr_v2_r50vd.weights.h5",
    },
    "rtdetr-v2-r101vd": {
        "model": "rtdetrv2_r101",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rt-detr-v2/rt_detr_v2_r101vd.weights.h5",
    },
}
