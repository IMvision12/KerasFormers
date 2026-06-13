COHERE_CONFIG = {
    "c4ai-command-r-v01": {
        "vocab_size": 256000,
        "embed_dim": 8192,
        "num_layers": 40,
        "num_heads": 64,
        "num_kv_heads": 64,
        "mlp_dim": 22528,
        "use_qk_norm": False,
        "norm_eps": 1e-5,
        "rope_theta": 8000000.0,
        "logit_scale": 0.0625,
        "tie_embeddings": True,
    },
    "c4ai-command-r-08-2024": {
        "vocab_size": 256000,
        "embed_dim": 8192,
        "num_layers": 40,
        "num_heads": 64,
        "num_kv_heads": 8,
        "mlp_dim": 24576,
        "use_qk_norm": False,
        "norm_eps": 1e-5,
        "rope_theta": 4000000.0,
        "logit_scale": 0.0625,
        "tie_embeddings": True,
    },
    "c4ai-command-r-plus": {
        "vocab_size": 256000,
        "embed_dim": 12288,
        "num_layers": 64,
        "num_heads": 96,
        "num_kv_heads": 8,
        "mlp_dim": 33792,
        "use_qk_norm": False,
        "norm_eps": 1e-5,
        "rope_theta": 75000000.0,
        "logit_scale": 0.8333333333333334,
        "tie_embeddings": True,
    },
}

COHERE_WEIGHTS_URLS = {
    "c4ai-command-r-v01": {
        "hf_id": "CohereForAI/c4ai-command-r-v01",
        "gated": True,
        "safetensors": True,
    },
    "c4ai-command-r-08-2024": {
        "hf_id": "CohereForAI/c4ai-command-r-08-2024",
        "gated": True,
        "safetensors": True,
    },
    "c4ai-command-r-plus": {
        "hf_id": "CohereForAI/c4ai-command-r-plus",
        "gated": True,
        "safetensors": True,
    },
}
