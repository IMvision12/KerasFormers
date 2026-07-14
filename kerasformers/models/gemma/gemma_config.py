GEMMA_CONFIG = {
    "gemma-2b": {
        "embed_dim": 2048,
        "mlp_dim": 16384,
        "num_layers": 18,
        "num_heads": 8,
        "num_kv_heads": 1,
    },
    "gemma-2b-it": {
        "embed_dim": 2048,
        "mlp_dim": 16384,
        "num_layers": 18,
        "num_heads": 8,
        "num_kv_heads": 1,
    },
    "gemma-1.1-2b-it": {
        "embed_dim": 2048,
        "mlp_dim": 16384,
        "num_layers": 18,
        "num_heads": 8,
        "num_kv_heads": 1,
    },
    "gemma-7b": {
        "embed_dim": 3072,
        "mlp_dim": 24576,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 16,
    },
    "gemma-7b-it": {
        "embed_dim": 3072,
        "mlp_dim": 24576,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 16,
    },
    "gemma-1.1-7b-it": {
        "embed_dim": 3072,
        "mlp_dim": 24576,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 16,
    },
}

GEMMA_WEIGHTS_URLS = {
    "gemma-2b": {"hf_id": "google/gemma-2b", "gated": True, "safetensors": True},
    "gemma-2b-it": {"hf_id": "google/gemma-2b-it", "gated": True, "safetensors": True},
    "gemma-1.1-2b-it": {
        "hf_id": "google/gemma-1.1-2b-it",
        "gated": True,
        "safetensors": True,
    },
    "gemma-7b": {"hf_id": "google/gemma-7b", "gated": True, "safetensors": True},
    "gemma-7b-it": {"hf_id": "google/gemma-7b-it", "gated": True, "safetensors": True},
    "gemma-1.1-7b-it": {
        "hf_id": "google/gemma-1.1-7b-it",
        "gated": True,
        "safetensors": True,
    },
}
