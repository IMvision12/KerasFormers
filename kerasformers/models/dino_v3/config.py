DINOV3_VIT_CONFIG = {
    "dinov3_vits16": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "use_swiglu": False,
        "num_register_tokens": 4,
        "layer_scale_init": 1.0,
        "rope_theta": 100.0,
    },
    "dinov3_vitb16": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "use_swiglu": False,
        "num_register_tokens": 4,
        "layer_scale_init": 1.0,
        "rope_theta": 100.0,
    },
    "dinov3_vitl16": {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "use_swiglu": False,
        "num_register_tokens": 4,
        "layer_scale_init": 1.0,
        "rope_theta": 100.0,
    },
}

DINOV3_VIT_WEIGHTS_URLS = {
    "dinov3_vits16": {
        "hf_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "gated": True,
    },
    "dinov3_vitb16": {
        "hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "gated": True,
    },
    "dinov3_vitl16": {
        "hf_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "gated": True,
    },
}

DINOV3_CONVNEXT_CONFIG = {
    "dinov3_convnext_tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dim": [96, 192, 384, 768],
    },
    "dinov3_convnext_small": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [96, 192, 384, 768],
    },
    "dinov3_convnext_base": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [128, 256, 512, 1024],
    },
    "dinov3_convnext_large": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [192, 384, 768, 1536],
    },
}

DINOV3_CONVNEXT_WEIGHTS_URLS = {
    "dinov3_convnext_tiny": {
        "hf_id": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "gated": True,
    },
    "dinov3_convnext_small": {
        "hf_id": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        "gated": True,
    },
    "dinov3_convnext_base": {
        "hf_id": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
        "gated": True,
    },
    "dinov3_convnext_large": {
        "hf_id": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
        "gated": True,
    },
}
