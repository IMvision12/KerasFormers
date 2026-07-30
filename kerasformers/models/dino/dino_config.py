DINO_VIT_CONFIG = {
    "dino_vits16": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
    },
    "dino_vits8": {
        "patch_size": 8,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
    },
    "dino_vitb16": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
    "dino_vitb8": {
        "patch_size": 8,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
}

DINO_VIT_WEIGHTS_URLS = {
    "dino_vits16": {
        "url": "https://huggingface.co/kerasformers/dino_vits16",
    },
    "dino_vits8": {
        "url": "https://huggingface.co/kerasformers/dino_vits8",
    },
    "dino_vitb16": {
        "url": "https://huggingface.co/kerasformers/dino_vitb16",
    },
    "dino_vitb8": {
        "url": "https://huggingface.co/kerasformers/dino_vitb8",
    },
}

DINO_RESNET_CONFIG = {
    "dino_resnet50": {
        "depths": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
    },
}

DINO_RESNET_WEIGHTS_URLS = {
    "dino_resnet50": {
        "url": "https://huggingface.co/kerasformers/dino_resnet50",
    },
}
