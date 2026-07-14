_COMMAND_R7B = {
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
}

_COMMAND_A = {
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
}

_TINY_AYA = {
    "vocab_size": 262144,
    "embed_dim": 2048,
    "num_layers": 36,
    "num_heads": 16,
    "num_kv_heads": 4,
    "head_dim": 128,
    "mlp_dim": 11008,
    "sliding_window": 4096,
    "sliding_window_pattern": 4,
    "norm_eps": 1e-5,
    "rope_theta": 50000.0,
    "logit_scale": 1.0,
    "tie_embeddings": True,
}

COHERE2_CONFIG = {
    "c4ai-command-r7b-12-2024": dict(_COMMAND_R7B),
    "c4ai-command-r7b-arabic-02-2025": dict(_COMMAND_R7B),
    "command-a-03-2025": dict(_COMMAND_A),
    "command-a-reasoning-08-2025": dict(_COMMAND_A),
    "command-a-translate-08-2025": dict(_COMMAND_A),
    "tiny-aya-base": dict(_TINY_AYA),
    "tiny-aya-earth": dict(_TINY_AYA),
    "tiny-aya-global": dict(_TINY_AYA),
    "tiny-aya-water": dict(_TINY_AYA),
}

COHERE2_WEIGHTS_URLS = {
    "c4ai-command-r7b-12-2024": {
        "hf_id": "CohereLabs/c4ai-command-r7b-12-2024",
        "gated": True,
        "safetensors": True,
    },
    "c4ai-command-r7b-arabic-02-2025": {
        "hf_id": "CohereLabs/c4ai-command-r7b-arabic-02-2025",
        "gated": True,
        "safetensors": True,
    },
    "command-a-03-2025": {
        "hf_id": "CohereLabs/c4ai-command-a-03-2025",
        "gated": True,
        "safetensors": True,
    },
    "command-a-reasoning-08-2025": {
        "hf_id": "CohereLabs/command-a-reasoning-08-2025",
        "gated": True,
        "safetensors": True,
    },
    "command-a-translate-08-2025": {
        "hf_id": "CohereLabs/command-a-translate-08-2025",
        "gated": True,
        "safetensors": True,
    },
    "tiny-aya-base": {
        "hf_id": "CohereLabs/tiny-aya-base",
        "gated": True,
        "safetensors": True,
    },
    "tiny-aya-earth": {
        "hf_id": "CohereLabs/tiny-aya-earth",
        "gated": True,
        "safetensors": True,
    },
    "tiny-aya-global": {
        "hf_id": "CohereLabs/tiny-aya-global",
        "gated": True,
        "safetensors": True,
    },
    "tiny-aya-water": {
        "hf_id": "CohereLabs/tiny-aya-water",
        "gated": True,
        "safetensors": True,
    },
}
