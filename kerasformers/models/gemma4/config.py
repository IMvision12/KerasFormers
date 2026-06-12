# Text-decoder configs from the official google/gemma-4-* checkpoints. The
# omnimodal towers (audio/vision) and the E2B/E4B efficient variants
# (per-layer inputs, shared KV) are not ported.
GEMMA4_12B = {
    "vocab_size": 262144,
    "embed_dim": 3840,
    "mlp_dim": 15360,
    "num_layers": 48,
    "num_heads": 16,
    "num_kv_heads": 8,
    "num_global_kv_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "k_eq_v": True,
    "enable_moe": False,
    "sliding_window": 1024,
    "sliding_window_pattern": 6,
    "partial_rotary_factor": 0.25,
    "final_logit_softcapping": 30.0,
    "norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_local_theta": 10000.0,
    "tie_embeddings": True,
}

GEMMA4_31B = {
    "vocab_size": 262144,
    "embed_dim": 5376,
    "mlp_dim": 21504,
    "num_layers": 60,
    "num_heads": 32,
    "num_kv_heads": 16,
    "num_global_kv_heads": 4,
    "head_dim": 256,
    "global_head_dim": 512,
    "k_eq_v": True,
    "enable_moe": False,
    "sliding_window": 1024,
    "sliding_window_pattern": 6,
    "partial_rotary_factor": 0.25,
    "final_logit_softcapping": 30.0,
    "norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_local_theta": 10000.0,
    "tie_embeddings": True,
}

GEMMA4_26B_A4B = {
    "vocab_size": 262144,
    "embed_dim": 2816,
    "mlp_dim": 2112,
    "num_layers": 30,
    "num_heads": 16,
    "num_kv_heads": 8,
    "num_global_kv_heads": 2,
    "head_dim": 256,
    "global_head_dim": 512,
    "k_eq_v": True,
    "enable_moe": True,
    "num_experts": 128,
    "top_k_experts": 8,
    "moe_mlp_dim": 704,
    "sliding_window": 1024,
    "sliding_window_pattern": 6,
    "partial_rotary_factor": 0.25,
    "final_logit_softcapping": 30.0,
    "norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "rope_local_theta": 10000.0,
    "tie_embeddings": True,
}

GEMMA4_CONFIG = {
    "gemma-4-12b": dict(GEMMA4_12B),
    "gemma-4-12b-it": dict(GEMMA4_12B),
    "gemma-4-31b-it": dict(GEMMA4_31B),
    "gemma-4-26b-a4b-it": dict(GEMMA4_26B_A4B),
}

GEMMA4_WEIGHTS_URLS = {
    "gemma-4-12b": {"hf_id": "google/gemma-4-12B", "gated": False, "safetensors": True},
    "gemma-4-12b-it": {
        "hf_id": "google/gemma-4-12B-it",
        "gated": False,
        "safetensors": True,
    },
    "gemma-4-31b-it": {
        "hf_id": "google/gemma-4-31B-it",
        "gated": False,
        "safetensors": True,
    },
    "gemma-4-26b-a4b-it": {
        "hf_id": "google/gemma-4-26B-A4B-it",
        "gated": False,
        "safetensors": True,
    },
}
