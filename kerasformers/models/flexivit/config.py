"""FlexiViT variant registry (timm-ported)."""

FLEXIVIT_MODEL_CONFIG = {
    "flexivit_small": {
        "patch_size": 16,
        "dim": 384,
        "depth": 12,
        "num_heads": 6,
        "no_embed_class": True,
        "image_size": 240,
        "num_classes": 1000,
    },
    "flexivit_base": {
        "patch_size": 16,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "no_embed_class": True,
        "image_size": 240,
        "num_classes": 1000,
    },
    "flexivit_base_in21k": {
        "patch_size": 16,
        "dim": 768,
        "depth": 12,
        "num_heads": 12,
        "no_embed_class": True,
        "image_size": 240,
        "num_classes": 21843,
    },
    "flexivit_large": {
        "patch_size": 16,
        "dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "no_embed_class": True,
        "image_size": 240,
        "num_classes": 1000,
    },
}

FLEXIVIT_WEIGHT_CONFIG = {
    "flexivit_small_1200ep_in1k": {
        "model": "flexivit_small",
        "timm_id": "flexivit_small.1200ep_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.2/flexivit_small_1200ep_in1k.weights.h5",
    },
    "flexivit_small_600ep_in1k": {
        "model": "flexivit_small",
        "timm_id": "flexivit_small.600ep_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.2/flexivit_small_600ep_in1k.weights.h5",
    },
    "flexivit_small_300ep_in1k": {
        "model": "flexivit_small",
        "timm_id": "flexivit_small.300ep_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.2/flexivit_small_300ep_in1k.weights.h5",
    },
    "flexivit_base_1200ep_in1k": {
        "model": "flexivit_base",
        "timm_id": "flexivit_base.1200ep_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.2/flexivit_base_1200ep_in1k.weights.h5",
    },
    "flexivit_base_300ep_in1k": {
        "model": "flexivit_base",
        "timm_id": "flexivit_base.300ep_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.2/flexivit_base_300ep_in1k.weights.h5",
    },
    "flexivit_base_1000ep_in21k": {
        "model": "flexivit_base_in21k",
        "timm_id": "flexivit_base.1000ep_in21k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.2/flexivit_base_1000ep_in21k.weights.h5",
    },
    "flexivit_base_300ep_in21k": {
        "model": "flexivit_base_in21k",
        "timm_id": "flexivit_base.300ep_in21k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.2/flexivit_base_300ep_in21k.weights.h5",
    },
    "flexivit_large_1200ep_in1k": {
        "model": "flexivit_large",
        "timm_id": "flexivit_large.1200ep_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.2/flexivit_large_1200ep_in1k.weights.h5",
    },
    "flexivit_large_600ep_in1k": {
        "model": "flexivit_large",
        "timm_id": "flexivit_large.600ep_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.2/flexivit_large_600ep_in1k.weights.h5",
    },
    "flexivit_large_300ep_in1k": {
        "model": "flexivit_large",
        "timm_id": "flexivit_large.300ep_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.2/flexivit_large_300ep_in1k.weights.h5",
    },
}
