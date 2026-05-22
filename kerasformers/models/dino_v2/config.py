DINOV2_CONFIG = {
    "dinov2_vits14": {
        "patch_size": 14,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "layer_scale_init": 1.0,
    },
    "dinov2_vitb14": {
        "patch_size": 14,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "layer_scale_init": 1.0,
    },
    "dinov2_vitl14": {
        "patch_size": 14,
        "dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "layer_scale_init": 1.0,
    },
}

DINOV2_WEIGHTS = {
    "dinov2_vits14": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/dino12/dinov2_vits14.weights.h5",
    },
    "dinov2_vitb14": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/dino12/dinov2_vitb14.weights.h5",
    },
    "dinov2_vitl14": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/dino12/dinov2_vitl14.weights.h5",
    },
}
