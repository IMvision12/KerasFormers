GEMMA2_2B = {
    "vocab_size": 256000,
    "embed_dim": 2304,
    "mlp_dim": 9216,
    "num_layers": 26,
    "num_heads": 8,
    "num_kv_heads": 4,
    "head_dim": 256,
    "query_pre_attn_scalar": 256.0,
    "attn_logit_softcapping": 50.0,
    "final_logit_softcapping": 30.0,
    "sliding_window": 4096,
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_embeddings": True,
}

GEMMA2_9B = {
    "vocab_size": 256000,
    "embed_dim": 3584,
    "mlp_dim": 14336,
    "num_layers": 42,
    "num_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 256,
    "query_pre_attn_scalar": 256.0,
    "attn_logit_softcapping": 50.0,
    "final_logit_softcapping": 30.0,
    "sliding_window": 4096,
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_embeddings": True,
}

GEMMA2_27B = {
    "vocab_size": 256000,
    "embed_dim": 4608,
    "mlp_dim": 36864,
    "num_layers": 46,
    "num_heads": 32,
    "num_kv_heads": 16,
    "head_dim": 128,
    "query_pre_attn_scalar": 144.0,
    "attn_logit_softcapping": 50.0,
    "final_logit_softcapping": 30.0,
    "sliding_window": 4096,
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_embeddings": True,
}

GEMMA2_CONFIG = {
    "gemma-2-2b": dict(GEMMA2_2B),
    "gemma-2-2b-it": dict(GEMMA2_2B),
    "gemma-2-9b": dict(GEMMA2_9B),
    "gemma-2-9b-it": dict(GEMMA2_9B),
    "gemma-2-27b": dict(GEMMA2_27B),
    "gemma-2-27b-it": dict(GEMMA2_27B),
}

GEMMA2_WEIGHTS_URLS = {
    "gemma-2-2b": {"hf_id": "google/gemma-2-2b", "gated": True, "safetensors": True},
    "gemma-2-2b-it": {
        "hf_id": "google/gemma-2-2b-it",
        "gated": True,
        "safetensors": True,
    },
    "gemma-2-9b": {"hf_id": "google/gemma-2-9b", "gated": True, "safetensors": True},
    "gemma-2-9b-it": {
        "hf_id": "google/gemma-2-9b-it",
        "gated": True,
        "safetensors": True,
    },
    "gemma-2-27b": {"hf_id": "google/gemma-2-27b", "gated": True, "safetensors": True},
    "gemma-2-27b-it": {
        "hf_id": "google/gemma-2-27b-it",
        "gated": True,
        "safetensors": True,
    },
}
