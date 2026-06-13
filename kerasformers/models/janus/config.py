JANUS_PRO_1B = {
    "vocab_size": 102400,
    "embed_dim": 2048,
    "mlp_dim": 5632,
    "num_layers": 24,
    "num_heads": 16,
    "num_kv_heads": 16,
    "head_dim": 128,
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_embeddings": False,
    "vision_embed_dim": 1024,
    "vision_mlp_dim": 4096,
    "vision_num_layers": 24,
    "vision_num_heads": 16,
    "image_size": 384,
    "patch_size": 16,
    "vision_norm_eps": 1e-6,
    "image_token_id": 100581,
}

JANUS_PRO_7B = {
    "vocab_size": 102400,
    "embed_dim": 4096,
    "mlp_dim": 11008,
    "num_layers": 30,
    "num_heads": 32,
    "num_kv_heads": 32,
    "head_dim": 128,
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_embeddings": False,
    "vision_embed_dim": 1024,
    "vision_mlp_dim": 4096,
    "vision_num_layers": 24,
    "vision_num_heads": 16,
    "image_size": 384,
    "patch_size": 16,
    "vision_norm_eps": 1e-6,
    "image_token_id": 100594,
}

JANUS_CONFIG = {
    "janus-pro-1b": dict(JANUS_PRO_1B),
    "janus-pro-7b": dict(JANUS_PRO_7B),
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
