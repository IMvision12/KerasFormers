RESMLP_MODEL_CONFIG = {
    "resmlp_12": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "mlp_ratio": 4,
        "init_values": 1e-4,
        "image_size": 224,
        "num_classes": 1000,
    },
    "resmlp_24": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 24,
        "mlp_ratio": 4,
        "init_values": 1e-5,
        "image_size": 224,
        "num_classes": 1000,
    },
    "resmlp_36": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 36,
        "mlp_ratio": 4,
        "init_values": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
    "resmlp_big_24": {
        "patch_size": 8,
        "embed_dim": 768,
        "depth": 24,
        "mlp_ratio": 4,
        "init_values": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
}

RESMLP_WEIGHT_CONFIG = {
    "resmlp_12_224_fb_in1k": {
        "model": "resmlp_12",
        "timm_id": "resmlp_12_224.fb_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resmlp_12_224_fb_in1k.weights.h5",
    },
    "resmlp_12_224_fb_distilled_in1k": {
        "model": "resmlp_12",
        "timm_id": "resmlp_12_224.fb_distilled_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resmlp_12_224_fb_distilled_in1k.weights.h5",
    },
    "resmlp_24_224_fb_in1k": {
        "model": "resmlp_24",
        "timm_id": "resmlp_24_224.fb_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resmlp_24_224_fb_in1k.weights.h5",
    },
    "resmlp_24_224_fb_distilled_in1k": {
        "model": "resmlp_24",
        "timm_id": "resmlp_24_224.fb_distilled_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resmlp_24_224_fb_distilled_in1k.weights.h5",
    },
    "resmlp_36_224_fb_in1k": {
        "model": "resmlp_36",
        "timm_id": "resmlp_36_224.fb_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resmlp_36_224_fb_in1k.weights.h5",
    },
    "resmlp_36_224_fb_distilled_in1k": {
        "model": "resmlp_36",
        "timm_id": "resmlp_36_224.fb_distilled_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resmlp_36_224_fb_distilled_in1k.weights.h5",
    },
    "resmlp_big_24_224_fb_in1k": {
        "model": "resmlp_big_24",
        "timm_id": "resmlp_big_24_224.fb_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resmlp_big_24_224_fb_in1k.weights.h5",
    },
    "resmlp_big_24_224_fb_distilled_in1k": {
        "model": "resmlp_big_24",
        "timm_id": "resmlp_big_24_224.fb_distilled_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resmlp_big_24_224_fb_distilled_in1k.weights.h5",
    },
    "resmlp_big_24_224_fb_in22k_ft_in1k": {
        "model": "resmlp_big_24",
        "timm_id": "resmlp_big_24_224.fb_in22k_ft_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resmlp_big_24_224_fb_in22k_ft_in1k.weights.h5",
    },
}
