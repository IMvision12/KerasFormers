DINO_VIT_CONFIG = {
    "dino_vits16": {
        "patch_size": 16,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
    },
    "dino_vits8": {
        "patch_size": 8,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
    },
    "dino_vitb16": {
        "patch_size": 16,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
    "dino_vitb8": {
        "patch_size": 8,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
}

DINO_VIT_WEIGHTS = {
    "dino_vits16": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/DinoV1_V2/dino_vits16.weights.h5",
    },
    "dino_vits8": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/DinoV1_V2/dino_vits8.weights.h5",
    },
    "dino_vitb16": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/DinoV1_V2/dino_vitb16.weights.h5",
    },
    "dino_vitb8": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/DinoV1_V2/dino_vitb8.weights.h5",
    },
}

DINO_RESNET_CONFIG = {
    "dino_resnet50": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
    },
}

DINO_RESNET_WEIGHTS = {
    "dino_resnet50": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/DinoV1_V2/dino_resnet50.weights.h5",
    },
}
