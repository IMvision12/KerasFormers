MOBILEVIT_MODEL_CONFIG = {
    "mobilevit_xxs": {
        "initial_dims": 16,
        "head_dims": 320,
        "block_dims": [16, 24, 48, 64, 80],
        "expansion_ratio": [2.0, 2.0, 2.0, 2.0, 2.0],
        "attention_dims": [None, None, 64, 80, 96],
        "image_size": 256,
        "num_classes": 1000,
    },
    "mobilevit_xs": {
        "initial_dims": 16,
        "head_dims": 384,
        "block_dims": [32, 48, 64, 80, 96],
        "expansion_ratio": [4.0, 4.0, 4.0, 4.0, 4.0],
        "attention_dims": [None, None, 96, 120, 144],
        "image_size": 256,
        "num_classes": 1000,
    },
    "mobilevit_s": {
        "initial_dims": 16,
        "head_dims": 640,
        "block_dims": [32, 64, 96, 128, 160],
        "expansion_ratio": [4.0, 4.0, 4.0, 4.0, 4.0],
        "attention_dims": [None, None, 144, 192, 240],
        "image_size": 256,
        "num_classes": 1000,
    },
}

MOBILEVIT_WEIGHT_CONFIG = {
    "mobilevit_xxs_cvnets_in1k": {
        "model": "mobilevit_xxs",
        "timm_id": "mobilevit_xxs.cvnets_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/mobilevit_xxs_cvnets_in1k.weights.h5",
    },
    "mobilevit_xs_cvnets_in1k": {
        "model": "mobilevit_xs",
        "timm_id": "mobilevit_xs.cvnets_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/mobilevit_xs_cvnets_in1k.weights.h5",
    },
    "mobilevit_s_cvnets_in1k": {
        "model": "mobilevit_s",
        "timm_id": "mobilevit_s.cvnets_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/mobilevit_s_cvnets_in1k.weights.h5",
    },
}
