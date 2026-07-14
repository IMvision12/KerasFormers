QWEN3_NEXT_80B = {
    "vocab_size": 151936,
    "embed_dim": 2048,
    "mlp_dim": 5120,
    "num_layers": 48,
    "num_heads": 16,
    "num_kv_heads": 2,
    "head_dim": 256,
    "norm_eps": 1e-6,
    "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25,
    "tie_embeddings": False,
    "full_attention_interval": 4,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_mlp_dim": 512,
    "shared_mlp_dim": 512,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": (),
}

QWEN3_5_MOE_CONFIG = {
    "qwen3-next-80b-a3b-instruct": dict(QWEN3_NEXT_80B),
    "qwen3-next-80b-a3b-thinking": dict(QWEN3_NEXT_80B),
}

QWEN3_5_MOE_WEIGHTS_URLS = {
    "qwen3-next-80b-a3b-instruct": {
        "hf_id": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "gated": False,
        "safetensors": True,
    },
    "qwen3-next-80b-a3b-thinking": {
        "hf_id": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "gated": False,
        "safetensors": True,
    },
}
