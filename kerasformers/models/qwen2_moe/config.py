QWEN2_MOE_CONFIG = {
    "qwen1.5-moe-a2.7b": {
        "vocab_size": 151936,
        "embed_dim": 2048,
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 16,
        "mlp_dim": 5632,
        "num_experts": 60,
        "num_experts_per_tok": 4,
        "moe_mlp_dim": 1408,
        "shared_mlp_dim": 5632,
        "norm_topk_prob": False,
        "decoder_sparse_step": 1,
        "mlp_only_layers": (),
        "rope_theta": 1000000.0,
        "norm_eps": 1e-6,
        "tie_embeddings": False,
    },
    "qwen2-57b-a14b": {
        "vocab_size": 151936,
        "embed_dim": 3584,
        "num_layers": 28,
        "num_heads": 28,
        "num_kv_heads": 4,
        "mlp_dim": 18944,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "moe_mlp_dim": 2560,
        "shared_mlp_dim": 20480,
        "norm_topk_prob": False,
        "decoder_sparse_step": 1,
        "mlp_only_layers": (),
        "rope_theta": 1000000.0,
        "norm_eps": 1e-6,
        "tie_embeddings": False,
    },
}

QWEN2_MOE_WEIGHTS_URLS = {
    "qwen1.5-moe-a2.7b": {
        "hf_id": "Qwen/Qwen1.5-MoE-A2.7B",
        "gated": False,
        "safetensors": True,
    },
    "qwen1.5-moe-a2.7b-chat": {
        "hf_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "gated": False,
        "safetensors": True,
    },
    "qwen2-57b-a14b": {
        "hf_id": "Qwen/Qwen2-57B-A14B",
        "gated": False,
        "safetensors": True,
    },
    "qwen2-57b-a14b-instruct": {
        "hf_id": "Qwen/Qwen2-57B-A14B-Instruct",
        "gated": False,
        "safetensors": True,
    },
}
QWEN2_MOE_CONFIG["qwen1.5-moe-a2.7b-chat"] = dict(QWEN2_MOE_CONFIG["qwen1.5-moe-a2.7b"])
QWEN2_MOE_CONFIG["qwen2-57b-a14b-instruct"] = dict(QWEN2_MOE_CONFIG["qwen2-57b-a14b"])
