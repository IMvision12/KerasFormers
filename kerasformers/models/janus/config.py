JANUS_CONFIG = {
    "janus-pro-1b": {
        "embed_dim": 2048,
        "mlp_dim": 5632,
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 16,
        "image_token_id": 100581,
    },
    "janus-pro-7b": {
        "embed_dim": 4096,
        "mlp_dim": 11008,
        "num_layers": 30,
        "num_heads": 32,
        "num_kv_heads": 32,
        "image_token_id": 100594,
    },
}

JANUS_WEIGHTS_URLS = {
    "janus-pro-1b": {
        "hf_id": "deepseek-community/Janus-Pro-1B",
        "gated": False,
        "safetensors": True,
    },
    "janus-pro-7b": {
        "hf_id": "deepseek-community/Janus-Pro-7B",
        "gated": False,
        "safetensors": True,
    },
}
