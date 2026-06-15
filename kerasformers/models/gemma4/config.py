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
        "top_k_experts": 8,
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
