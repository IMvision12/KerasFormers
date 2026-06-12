GEMMA_2B = {
    "vocab_size": 256000,
    "embed_dim": 2048,
    "mlp_dim": 16384,
    "num_layers": 18,
    "num_heads": 8,
    "num_kv_heads": 1,
    "head_dim": 256,
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_embeddings": True,
}

GEMMA_7B = {
    "vocab_size": 256000,
    "embed_dim": 3072,
    "mlp_dim": 24576,
    "num_layers": 28,
    "num_heads": 16,
    "num_kv_heads": 16,
    "head_dim": 256,
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_embeddings": True,
}

GEMMA_CONFIG = {
    "gemma-2b": dict(GEMMA_2B),
    "gemma-2b-it": dict(GEMMA_2B),
    "gemma-1.1-2b-it": dict(GEMMA_2B),
    "gemma-7b": dict(GEMMA_7B),
    "gemma-7b-it": dict(GEMMA_7B),
    "gemma-1.1-7b-it": dict(GEMMA_7B),
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
