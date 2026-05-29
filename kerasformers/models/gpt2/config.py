GPT2_CONFIG = {
    "gpt2": {
        "vocab_size": 50257,
        "embed_dim": 768,
        "mlp_dim": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
    "gpt2-medium": {
        "vocab_size": 50257,
        "embed_dim": 1024,
        "mlp_dim": 4096,
        "num_layers": 24,
        "num_heads": 16,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
    "gpt2-large": {
        "vocab_size": 50257,
        "embed_dim": 1280,
        "mlp_dim": 5120,
        "num_layers": 36,
        "num_heads": 20,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
    "gpt2-xl": {
        "vocab_size": 50257,
        "embed_dim": 1600,
        "mlp_dim": 6400,
        "num_layers": 48,
        "num_heads": 25,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
}

GPT2_WEIGHTS = {
    "gpt2": {"hf_id": "openai-community/gpt2", "gated": False, "safetensors": True},
    "gpt2-medium": {
        "hf_id": "openai-community/gpt2-medium",
        "gated": False,
        "safetensors": True,
    },
    "gpt2-large": {
        "hf_id": "openai-community/gpt2-large",
        "gated": False,
        "safetensors": True,
    },
    "gpt2-xl": {
        "hf_id": "openai-community/gpt2-xl",
        "gated": False,
        "safetensors": True,
    },
}
