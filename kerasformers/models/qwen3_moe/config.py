QWEN3_MOE_BASE = {
    "vocab_size": 151936,
    "embed_dim": 2048,
    "num_layers": 48,
    "num_heads": 32,
    "num_kv_heads": 4,
    "head_dim": 128,
    "mlp_dim": 6144,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_mlp_dim": 768,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": (),
    "rope_theta": 1000000.0,
    "norm_eps": 1e-6,
    "tie_embeddings": False,
}

QWEN3_MOE_CONFIG = {
    "qwen3-30b-a3b": dict(QWEN3_MOE_BASE),
    "qwen3-30b-a3b-instruct-2507": dict(QWEN3_MOE_BASE),
    "qwen3-235b-a22b": {
        **QWEN3_MOE_BASE,
        "embed_dim": 4096,
        "num_layers": 94,
        "num_heads": 64,
        "num_kv_heads": 4,
        "mlp_dim": 12288,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_mlp_dim": 1536,
    },
}

QWEN3_MOE_WEIGHTS_URLS = {
    "qwen3-30b-a3b": {
        "hf_id": "Qwen/Qwen3-30B-A3B",
        "gated": False,
        "safetensors": True,
    },
    "qwen3-30b-a3b-instruct-2507": {
        "hf_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "gated": False,
        "safetensors": True,
    },
    "qwen3-235b-a22b": {
        "hf_id": "Qwen/Qwen3-235B-A22B",
        "gated": False,
        "safetensors": True,
    },
}
