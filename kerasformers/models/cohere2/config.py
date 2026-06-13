COHERE2_CONFIG = {
    "c4ai-command-r7b-12-2024": {
        "vocab_size": 256000,
        "embed_dim": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "mlp_dim": 14336,
        "sliding_window": 4096,
        "sliding_window_pattern": 4,
        "norm_eps": 1e-5,
        "rope_theta": 50000.0,
        "logit_scale": 0.25,
        "tie_embeddings": True,
    },
    "command-a-03-2025": {
        "vocab_size": 256000,
        "embed_dim": 12288,
        "num_layers": 64,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_dim": 128,
        "mlp_dim": 36864,
        "sliding_window": 4096,
        "sliding_window_pattern": 4,
        "norm_eps": 1e-5,
        "rope_theta": 50000.0,
        "logit_scale": 0.25,
        "tie_embeddings": True,
    },
}

COHERE2_WEIGHTS_URLS = {
    "c4ai-command-r7b-12-2024": {
        "hf_id": "CohereLabs/c4ai-command-r7b-12-2024",
        "gated": True,
        "safetensors": True,
    },
    "command-a-03-2025": {
        "hf_id": "CohereLabs/c4ai-command-a-03-2025",
        "gated": True,
        "safetensors": True,
    },
}
