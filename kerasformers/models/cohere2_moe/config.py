COHERE2_MOE_CONFIG = {
    "command-moe": {
        "vocab_size": 256000,
        "embed_dim": 8192,
        "num_layers": 40,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "mlp_dim": 22528,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "expert_selection_fn": "softmax",
        "norm_topk_prob": True,
        "num_shared_experts": 0,
        "shared_combine": "average",
        "first_k_dense_replace": 0,
        "sliding_window": 4096,
        "sliding_window_pattern": 4,
        "rms_norm_eps": None,
        "norm_eps": 1e-5,
        "rope_theta": 50000.0,
        "logit_scale": 0.0625,
        "tie_embeddings": True,
    },
}

COHERE2_MOE_WEIGHTS_URLS = {}
