GLM_CONFIG = {
    "glm-4-9b": {
        "vocab_size": 151552,
        "embed_dim": 4096,
        "num_layers": 40,
        "num_heads": 32,
        "num_kv_heads": 2,
        "head_dim": 128,
        "mlp_dim": 13696,
        "partial_rotary_factor": 0.5,
        "norm_eps": 0.00000015625,
        "rope_theta": 10000.0,
        "attention_bias": True,
        "tie_embeddings": False,
    },
    "glm-4-9b-chat": {
        "vocab_size": 151552,
        "embed_dim": 4096,
        "num_layers": 40,
        "num_heads": 32,
        "num_kv_heads": 2,
        "head_dim": 128,
        "mlp_dim": 13696,
        "partial_rotary_factor": 0.5,
        "norm_eps": 0.00000015625,
        "rope_theta": 10000.0,
        "attention_bias": True,
        "tie_embeddings": False,
    },
}

GLM_WEIGHTS_URLS = {
    "glm-4-9b": {
        "hf_id": "THUDM/glm-4-9b",
        "gated": True,
        "safetensors": True,
    },
    "glm-4-9b-chat": {
        "hf_id": "THUDM/glm-4-9b-chat-hf",
        "gated": True,
        "safetensors": True,
    },
}
