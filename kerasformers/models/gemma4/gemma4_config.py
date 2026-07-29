GEMMA4_CONFIG = {
    "gemma-4-12b": {
        "embed_dim": 3840,
        "mlp_dim": 15360,
        "num_layers": 48,
        "num_heads": 16,
        "num_kv_heads": 8,
        "num_global_kv_heads": 1,
        "enable_moe": False,
    },
    "gemma-4-12b-it": {
        "embed_dim": 3840,
        "mlp_dim": 15360,
        "num_layers": 48,
        "num_heads": 16,
        "num_kv_heads": 8,
        "num_global_kv_heads": 1,
        "enable_moe": False,
    },
    "gemma-4-31b-it": {
        "embed_dim": 5376,
        "mlp_dim": 21504,
        "num_layers": 60,
        "num_heads": 32,
        "num_kv_heads": 16,
        "num_global_kv_heads": 4,
        "enable_moe": False,
    },
    "gemma-4-26b-a4b-it": {
        "embed_dim": 2816,
        "mlp_dim": 2112,
        "num_layers": 30,
        "num_heads": 16,
        "num_kv_heads": 8,
        "num_global_kv_heads": 2,
        "enable_moe": True,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_mlp_dim": 704,
    },
}

GEMMA4_WEIGHTS_URLS = {
    "gemma-4-12b": {"hf_id": "google/gemma-4-12B", "gated": False, "safetensors": True},
    "gemma-4-12b-it": {
        "hf_id": "google/gemma-4-12B-it",
        "gated": False,
        "safetensors": True,
    },
    "gemma-4-31b-it": {
        "hf_id": "google/gemma-4-31B-it",
        "gated": False,
        "safetensors": True,
    },
    "gemma-4-26b-a4b-it": {
        "hf_id": "google/gemma-4-26B-A4B-it",
        "gated": False,
        "safetensors": True,
    },
}

# The vision-carrying gemma4 checkpoints (model_type "gemma4") share one NaViT
# tower. The 31B and 26B-A4B checkpoints carry a vision tower but no audio tower;
# gemma-4-12B is the separate "gemma4_unified" architecture and is text-only here.
GEMMA4_VISION_CONFIG = {
    "hidden_size": 1152,
    "num_layers": 27,
    "num_heads": 16,
    "num_kv_heads": 16,
    "head_dim": 72,
    "intermediate_size": 4304,
    "patch_size": 16,
    "position_embedding_size": 10240,
    "pooling_kernel_size": 3,
    "rope_theta": 100.0,
    "eps": 1e-6,
    "standardize": True,
    "use_clipped_linears": False,
}

GEMMA4_MULTIMODAL_CONFIG = {
    "gemma-4-31b-it": {
        "text_config": {
            "embed_dim": 5376,
            "mlp_dim": 21504,
            "num_layers": 60,
            "num_heads": 32,
            "num_kv_heads": 16,
            "num_global_kv_heads": 4,
            "enable_moe": False,
        },
        "vision_config": dict(GEMMA4_VISION_CONFIG),
        "image_token_id": 258880,
        "video_token_id": 258884,
        "audio_token_id": 258881,
        "use_bidirectional_vision": True,
    },
    "gemma-4-26b-a4b-it": {
        "text_config": {
            "embed_dim": 2816,
            "mlp_dim": 2112,
            "num_layers": 30,
            "num_heads": 16,
            "num_kv_heads": 8,
            "num_global_kv_heads": 2,
            "enable_moe": True,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_mlp_dim": 704,
        },
        "vision_config": dict(GEMMA4_VISION_CONFIG),
        "image_token_id": 258880,
        "video_token_id": 258884,
        "audio_token_id": 258881,
        "use_bidirectional_vision": True,
    },
}

GEMMA4_MULTIMODAL_WEIGHTS_URLS = {
    "gemma-4-31b-it": {
        "hf_id": "google/gemma-4-31B-it",
        "gated": False,
        "safetensors": True,
    },
    "gemma-4-26b-a4b-it": {
        "hf_id": "google/gemma-4-26B-A4B-it",
        "gated": False,
        "safetensors": True,
    },
}
