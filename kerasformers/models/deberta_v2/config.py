DEBERTA_V2_MODEL_CONFIG = {
    "deberta_v2_xlarge": {
        "vocab_size": 128100,
        "embed_dim": 1536,
        "num_layers": 24,
        "num_heads": 24,
        "mlp_dim": 6144,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "position_buckets": 256,
        "pos_att_type": ["p2c", "c2p"],
        "norm_rel_ebd": True,
        "conv_kernel_size": 3,
        "conv_act": "gelu",
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-7,
        "pad_token_id": 0,
    },
    "deberta_v2_xxlarge": {
        "vocab_size": 128100,
        "embed_dim": 1536,
        "num_layers": 48,
        "num_heads": 24,
        "mlp_dim": 6144,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "position_buckets": 256,
        "pos_att_type": ["p2c", "c2p"],
        "norm_rel_ebd": True,
        "conv_kernel_size": 3,
        "conv_act": "gelu",
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-7,
        "pad_token_id": 0,
    },
}

DEBERTA_V2_WEIGHT_CONFIG = {
    "deberta_v2_xlarge": {
        "model": "deberta_v2_xlarge",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/deberta_v2/deberta_v2_xlarge.weights.json",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/deberta_v2/deberta_v2_xlarge_mlm.weights.json",
    },
    "deberta_v2_xxlarge": {
        "model": "deberta_v2_xxlarge",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/deberta_v2/deberta_v2_xxlarge.weights.json",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/deberta_v2/deberta_v2_xxlarge_mlm.weights.json",
    },
}

DEBERTA_V2_VOCAB_URL = "https://github.com/IMvision12/KerasFormers/releases/download/deberta_v2/deberta_v2_spm.model"
