YARN_V4 = {
    "type": "yarn",
    "factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "original_max_position_embeddings": 65536,
}

V4_FLASH = {
    "vocab_size": 129280,
    "embed_dim": 4096,
    "num_layers": 43,
    "num_heads": 64,
    "head_dim": 512,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "layer_types": tuple(
        ["sliding_attention"] * 2
        + [
            "compressed_sparse_attention"
            if i % 2 == 0
            else "heavily_compressed_attention"
            for i in range(41)
        ]
    ),
    "mlp_layer_types": tuple(["hash_moe"] * 3 + ["moe"] * 40),
    "num_experts": 256,
    "num_experts_per_tok": 6,
    "moe_mlp_dim": 2048,
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,
    "sliding_window": 128,
    "compress_rate_csa": 4,
    "compress_rate_hca": 128,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 512,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "rope_theta": 10000.0,
    "compress_rope_theta": 160000.0,
    "rope_scaling": dict(YARN_V4),
    "norm_eps": 1e-6,
    "tie_embeddings": False,
}

V4_PRO = {
    **V4_FLASH,
    "embed_dim": 7168,
    "num_layers": 61,
    "num_heads": 128,
    "o_groups": 16,
    "num_experts": 384,
    "moe_mlp_dim": 3072,
    "routed_scaling_factor": 2.5,
    "index_topk": 1024,
    "layer_types": None,
    "mlp_layer_types": None,
}

DEEPSEEK_V4_CONFIG = {
    "deepseek-v4-flash": dict(V4_FLASH),
    "deepseek-v4-flash-base": dict(V4_FLASH),
    "deepseek-v4-pro": dict(V4_PRO),
    "deepseek-v4-pro-base": dict(V4_PRO),
}

DEEPSEEK_V4_WEIGHTS_URLS = {
    "deepseek-v4-flash": {
        "hf_id": "deepseek-ai/DeepSeek-V4-Flash",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v4-flash-base": {
        "hf_id": "deepseek-ai/DeepSeek-V4-Flash-Base",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v4-pro": {
        "hf_id": "deepseek-ai/DeepSeek-V4-Pro",
        "gated": False,
        "safetensors": True,
    },
    "deepseek-v4-pro-base": {
        "hf_id": "deepseek-ai/DeepSeek-V4-Pro-Base",
        "gated": False,
        "safetensors": True,
    },
}
