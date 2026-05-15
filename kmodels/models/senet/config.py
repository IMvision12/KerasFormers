SENET_MODEL_CONFIG = {
    "seresnet50_a1_in1k": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "timm_id": "seresnet50.a1_in1k",
        "senet": True,
    },
    "seresnext50_32x4d_racm_in1k": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "timm_id": "seresnext50_32x4d.racm_in1k",
        "senet": True,
        "block_fn_name": "resnext_block",
    },
    "seresnext50_32x4d_gluon_in1k": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "timm_id": "seresnext50_32x4d.gluon_in1k",
        "senet": True,
        "block_fn_name": "resnext_block",
    },
    "seresnext101_32x4d_gluon_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "timm_id": "seresnext101_32x4d.gluon_in1k",
        "senet": True,
        "block_fn_name": "resnext_block",
    },
    "seresnext101_32x8d_ah_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 4,
        "timm_id": "seresnext101_32x8d.ah_in1k",
        "senet": True,
        "block_fn_name": "resnext_block",
    },
}

SENET_WEIGHT_CONFIG = {
    "seresnet50_a1_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/seresnet50_a1_in1k.weights.h5",
    },
    "seresnext50_32x4d_racm_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/seresnext50_32x4d_racm_in1k.weights.h5",
    },
    "seresnext50_32x4d_gluon_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/seresnext50_32x4d_gluon_in1k.weights.h5",
    },
    "seresnext101_32x4d_gluon_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/seresnext101_32x4d_gluon_in1k.weights.h5",
    },
    "seresnext101_32x8d_ah_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/seresnext101_32x8d_ah_in1k.weights.h5",
    },
}
