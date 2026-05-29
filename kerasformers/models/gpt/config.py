GPT_CONFIG = {
    "gpt": {
        "vocab_size": 40478,
        "embed_dim": 768,
        "mlp_dim": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 512,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
}

GPT_WEIGHTS = {
    "gpt": {
        "hf_id": "openai-community/openai-gpt",
        "gated": False,
        "safetensors": True,
    },
}
