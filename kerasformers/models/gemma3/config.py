VISION_SO400M = {
    "vision_embed_dim": 1152,
    "vision_mlp_dim": 4304,
    "vision_num_layers": 27,
    "vision_num_heads": 16,
    "image_size": 896,
    "patch_size": 14,
    "vision_norm_eps": 1e-6,
    "mm_tokens_per_image": 256,
    "image_token_id": 262144,
}

GEMMA3_1B = {
    "vocab_size": 262144,
    "embed_dim": 1152,
    "mlp_dim": 6912,
    "num_layers": 26,
    "num_heads": 4,
    "num_kv_heads": 1,
    "head_dim": 256,
    "query_pre_attn_scalar": 256.0,
    "sliding_window": 512,
    "sliding_window_pattern": 6,
    "norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_local_theta": 10000.0,
    "rope_scaling_factor": None,
    "tie_embeddings": True,
    "vision_num_layers": 0,
}

GEMMA3_4B = {
    "vocab_size": 262208,
    "embed_dim": 2560,
    "mlp_dim": 10240,
    "num_layers": 34,
    "num_heads": 8,
    "num_kv_heads": 4,
    "head_dim": 256,
    "query_pre_attn_scalar": 256.0,
    "sliding_window": 1024,
    "sliding_window_pattern": 6,
    "norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_local_theta": 10000.0,
    "rope_scaling_factor": 8.0,
    "tie_embeddings": True,
    **VISION_SO400M,
}

GEMMA3_12B = {
    "vocab_size": 262208,
    "embed_dim": 3840,
    "mlp_dim": 15360,
    "num_layers": 48,
    "num_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 256,
    "query_pre_attn_scalar": 256.0,
    "sliding_window": 1024,
    "sliding_window_pattern": 6,
    "norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_local_theta": 10000.0,
    "rope_scaling_factor": 8.0,
    "tie_embeddings": True,
    **VISION_SO400M,
}

GEMMA3_27B = {
    "vocab_size": 262208,
    "embed_dim": 5376,
    "mlp_dim": 21504,
    "num_layers": 62,
    "num_heads": 32,
    "num_kv_heads": 16,
    "head_dim": 128,
    "query_pre_attn_scalar": 168.0,
    "sliding_window": 1024,
    "sliding_window_pattern": 6,
    "norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_local_theta": 10000.0,
    "rope_scaling_factor": 8.0,
    "tie_embeddings": True,
    **VISION_SO400M,
}

GEMMA3_CONFIG = {
    "gemma-3-1b-pt": dict(GEMMA3_1B),
    "gemma-3-1b-it": dict(GEMMA3_1B),
    "gemma-3-4b-pt": dict(GEMMA3_4B),
    "gemma-3-4b-it": dict(GEMMA3_4B),
    "gemma-3-12b-pt": dict(GEMMA3_12B),
    "gemma-3-12b-it": dict(GEMMA3_12B),
    "gemma-3-27b-pt": dict(GEMMA3_27B),
    "gemma-3-27b-it": dict(GEMMA3_27B),
}

GEMMA3_WEIGHTS_URLS = {
    "gemma-3-1b-pt": {
        "hf_id": "google/gemma-3-1b-pt",
        "gated": True,
        "safetensors": True,
    },
    "gemma-3-1b-it": {
        "hf_id": "google/gemma-3-1b-it",
        "gated": True,
        "safetensors": True,
    },
    "gemma-3-4b-pt": {
        "hf_id": "google/gemma-3-4b-pt",
        "gated": True,
        "safetensors": True,
    },
    "gemma-3-4b-it": {
        "hf_id": "google/gemma-3-4b-it",
        "gated": True,
        "safetensors": True,
    },
    "gemma-3-12b-pt": {
        "hf_id": "google/gemma-3-12b-pt",
        "gated": True,
        "safetensors": True,
    },
    "gemma-3-12b-it": {
        "hf_id": "google/gemma-3-12b-it",
        "gated": True,
        "safetensors": True,
    },
    "gemma-3-27b-pt": {
        "hf_id": "google/gemma-3-27b-pt",
        "gated": True,
        "safetensors": True,
    },
    "gemma-3-27b-it": {
        "hf_id": "google/gemma-3-27b-it",
        "gated": True,
        "safetensors": True,
    },
}
