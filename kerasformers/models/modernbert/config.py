MODERNBERT_MODEL_CONFIG = {
    "modernbert_base": {
        "vocab_size": 50368,
        "embed_dim": 768,
        "num_layers": 22,
        "num_heads": 12,
        "mlp_dim": 1152,
        "max_position_embeddings": 8192,
        "hidden_act": "gelu",
        "norm_eps": 1e-5,
        "local_attention": 128,
        "global_attn_every_n_layers": 3,
        "global_rope_theta": 160000.0,
        "local_rope_theta": 10000.0,
        "pad_token_id": 50283,
    },
    "modernbert_large": {
        "vocab_size": 50368,
        "embed_dim": 1024,
        "num_layers": 28,
        "num_heads": 16,
        "mlp_dim": 2624,
        "max_position_embeddings": 8192,
        "hidden_act": "gelu",
        "norm_eps": 1e-5,
        "local_attention": 128,
        "global_attn_every_n_layers": 3,
        "global_rope_theta": 160000.0,
        "local_rope_theta": 10000.0,
        "pad_token_id": 50283,
    },
}

MODERNBERT_WEIGHT_CONFIG = {
    "modernbert_base": {
        "model": "modernbert_base",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/modernbert/modernbert_base.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/modernbert/modernbert_base_mlm.weights.h5",
    },
    "modernbert_large": {
        "model": "modernbert_large",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/modernbert/modernbert_large.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/modernbert/modernbert_large_mlm.weights.h5",
    },
}

MODERNBERT_VOCAB_URL = "https://github.com/IMvision12/KerasFormers/releases/download/modernbert/modernbert_tokenizer.json"
