SENET_MODEL_CONFIG = {
    "seresnet50": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "senet": True,
    },
    "seresnext50_32x4d": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "senet": True,
        "block_fn_name": "resnext_block",
    },
    "seresnext101_32x4d": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "senet": True,
        "block_fn_name": "resnext_block",
    },
    "seresnext101_32x8d": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 4,
        "senet": True,
        "block_fn_name": "resnext_block",
    },
}

SENET_WEIGHT_CONFIG = {
    "seresnet50_a1_in1k": {
        "model": "seresnet50",
        "timm_id": "seresnet50.a1_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/seresnet50_a1_in1k.weights.h5",
    },
    "seresnext50_32x4d_racm_in1k": {
        "model": "seresnext50_32x4d",
        "timm_id": "seresnext50_32x4d.racm_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/seresnext50_32x4d_racm_in1k.weights.h5",
    },
    "seresnext50_32x4d_gluon_in1k": {
        "model": "seresnext50_32x4d",
        "timm_id": "seresnext50_32x4d.gluon_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/seresnext50_32x4d_gluon_in1k.weights.h5",
    },
    "seresnext101_32x4d_gluon_in1k": {
        "model": "seresnext101_32x4d",
        "timm_id": "seresnext101_32x4d.gluon_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/seresnext101_32x4d_gluon_in1k.weights.h5",
    },
    "seresnext101_32x8d_ah_in1k": {
        "model": "seresnext101_32x8d",
        "timm_id": "seresnext101_32x8d.ah_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/seresnext101_32x8d_ah_in1k.weights.h5",
    },
}
