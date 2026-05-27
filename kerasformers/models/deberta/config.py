DEBERTA_MODEL_CONFIG = {
    "deberta_base": {
        "vocab_size": 50265,
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "pos_att_type": ["c2p", "p2c"],
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-7,
        "pad_token_id": 0,
    },
    "deberta_large": {
        "vocab_size": 50265,
        "embed_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "mlp_dim": 4096,
        "max_position_embeddings": 512,
        "max_relative_positions": 512,
        "pos_att_type": ["c2p", "p2c"],
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-7,
        "pad_token_id": 0,
    },
}

DEBERTA_WEIGHT_CONFIG = {
    "deberta_base": {
        "model": "deberta_base",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/deberta/deberta_base.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/deberta/deberta_base_mlm.weights.h5",
    },
    "deberta_large": {
        "model": "deberta_large",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/deberta/deberta_large.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/deberta/deberta_large_mlm.weights.h5",
    },
}

DEBERTA_VOCAB_URL = "https://github.com/IMvision12/KerasFormers/releases/download/deberta/deberta_vocab.json"
DEBERTA_MERGES_URL = "https://github.com/IMvision12/KerasFormers/releases/download/deberta/deberta_merges.txt"
