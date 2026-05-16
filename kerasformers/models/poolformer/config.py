POOLFORMER_MODEL_CONFIG = {
    "poolformer_s12": {
        "embed_dims": (64, 128, 320, 512),
        "num_blocks": (2, 2, 6, 2),
        "init_scale": 1e-5,
        "image_size": 224,
        "num_classes": 1000,
    },
    "poolformer_s24": {
        "embed_dims": (64, 128, 320, 512),
        "num_blocks": (4, 4, 12, 4),
        "init_scale": 1e-5,
        "image_size": 224,
        "num_classes": 1000,
    },
    "poolformer_s36": {
        "embed_dims": (64, 128, 320, 512),
        "num_blocks": (6, 6, 18, 6),
        "init_scale": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
    "poolformer_m36": {
        "embed_dims": (96, 192, 384, 768),
        "num_blocks": (6, 6, 18, 6),
        "init_scale": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
    "poolformer_m48": {
        "embed_dims": (96, 192, 384, 768),
        "num_blocks": (8, 8, 24, 8),
        "init_scale": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
}

POOLFORMER_WEIGHT_CONFIG = {
    "poolformer_s12_sail_in1k": {
        "model": "poolformer_s12",
        "timm_id": "poolformer_s12.sail_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/poolformer_s12_sail_in1k.weights.h5",
    },
    "poolformer_s24_sail_in1k": {
        "model": "poolformer_s24",
        "timm_id": "poolformer_s24.sail_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/poolformer_s24_sail_in1k.weights.h5",
    },
    "poolformer_s36_sail_in1k": {
        "model": "poolformer_s36",
        "timm_id": "poolformer_s36.sail_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/poolformer_s36_sail_in1k.weights.h5",
    },
    "poolformer_m36_sail_in1k": {
        "model": "poolformer_m36",
        "timm_id": "poolformer_m36.sail_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/poolformer_m36_sail_in1k.weights.h5",
    },
    "poolformer_m48_sail_in1k": {
        "model": "poolformer_m48",
        "timm_id": "poolformer_m48.sail_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/poolformer_m48_sail_in1k.weights.h5",
    },
}
