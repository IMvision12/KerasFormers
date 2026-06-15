# Only the per-variant differences live here; everything common to all GLM-4
# checkpoints (vocab_size, num_kv_heads, head_dim, partial_rotary_factor,
# norm_eps, rope_theta, attention_bias, tie_embeddings) is baked into the
# Glm4Model.__init__ defaults.
GLM4_CONFIG = {
    "glm-4-9b-0414": {
        "embed_dim": 4096,
        "num_layers": 40,
        "num_heads": 32,
        "mlp_dim": 13696,
    },
    "glm-4-32b-0414": {
        "embed_dim": 6144,
        "num_layers": 61,
        "num_heads": 48,
        "mlp_dim": 23040,
    },
    "glm-z1-9b-0414": {
        "embed_dim": 4096,
        "num_layers": 40,
        "num_heads": 32,
        "mlp_dim": 13696,
    },
    "glm-z1-32b-0414": {
        "embed_dim": 6144,
        "num_layers": 61,
        "num_heads": 48,
        "mlp_dim": 23040,
    },
}

GLM4_WEIGHTS_URLS = {
    "glm-4-9b-0414": {
        "hf_id": "THUDM/GLM-4-9B-0414",
        "gated": False,
        "safetensors": True,
    },
    "glm-4-32b-0414": {
        "hf_id": "THUDM/GLM-4-32B-0414",
        "gated": False,
        "safetensors": True,
    },
    "glm-z1-9b-0414": {
        "hf_id": "THUDM/GLM-Z1-9B-0414",
        "gated": False,
        "safetensors": True,
    },
    "glm-z1-32b-0414": {
        "hf_id": "THUDM/GLM-Z1-32B-0414",
        "gated": False,
        "safetensors": True,
    },
}
