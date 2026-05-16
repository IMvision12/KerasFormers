MLP_MIXER_MODEL_CONFIG = {
    "mixer_b16_224_in21k": {
        "patch_size": 16,
        "num_blocks": 12,
        "embed_dim": 768,
        "mlp_ratio": (0.5, 4.0),
        "image_size": 224,
        "num_classes": 21843,
    },
    "mixer_b16_224": {
        "patch_size": 16,
        "num_blocks": 12,
        "embed_dim": 768,
        "mlp_ratio": (0.5, 4.0),
        "image_size": 224,
        "num_classes": 1000,
    },
    "mixer_l16_224_in21k": {
        "patch_size": 16,
        "num_blocks": 24,
        "embed_dim": 1024,
        "mlp_ratio": (0.5, 4.0),
        "image_size": 224,
        "num_classes": 21843,
    },
    "mixer_l16_224": {
        "patch_size": 16,
        "num_blocks": 24,
        "embed_dim": 1024,
        "mlp_ratio": (0.5, 4.0),
        "image_size": 224,
        "num_classes": 1000,
    },
}

MLP_MIXER_WEIGHT_CONFIG = {
    "mixer_b16_224_goog_in21k": {
        "model": "mixer_b16_224_in21k",
        "timm_id": "mixer_b16_224.goog_in21k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/mixer_b16_224_goog_in21k.weights.h5",
    },
    "mixer_b16_224_goog_in21k_ft_in1k": {
        "model": "mixer_b16_224",
        "timm_id": "mixer_b16_224.goog_in21k_ft_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/mixer_b16_224_goog_in21k_ft_in1k.weights.h5",
    },
    "mixer_b16_224_miil_in21k_ft_in1k": {
        "model": "mixer_b16_224",
        "timm_id": "mixer_b16_224.miil_in21k_ft_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/mixer_b16_224_miil_in21k_ft_in1k.weights.h5",
    },
    "mixer_l16_224_goog_in21k": {
        "model": "mixer_l16_224_in21k",
        "timm_id": "mixer_l16_224.goog_in21k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/mixer_l16_224_goog_in21k.weights.h5",
    },
    "mixer_l16_224_goog_in21k_ft_in1k": {
        "model": "mixer_l16_224",
        "timm_id": "mixer_l16_224.goog_in21k_ft_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/mixer_l16_224_goog_in21k_ft_in1k.weights.h5",
    },
}
