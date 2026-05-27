_COMMON = {
    "vocab_size": 128100,
    "max_position_embeddings": 512,
    "max_relative_positions": 512,
    "position_buckets": 256,
    "pos_att_type": ["p2c", "c2p"],
    "norm_rel_ebd": True,
    "conv_kernel_size": 0,
    "conv_act": "gelu",
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-7,
    "pad_token_id": 0,
}

DEBERTA_V3_MODEL_CONFIG = {
    "deberta_v3_xsmall": {
        **_COMMON,
        "embed_dim": 384,
        "num_layers": 12,
        "num_heads": 6,
        "mlp_dim": 1536,
    },
    "deberta_v3_small": {
        **_COMMON,
        "embed_dim": 768,
        "num_layers": 6,
        "num_heads": 12,
        "mlp_dim": 3072,
    },
    "deberta_v3_base": {
        **_COMMON,
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
    },
    "deberta_v3_large": {
        **_COMMON,
        "embed_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "mlp_dim": 4096,
    },
}

_REL = "https://github.com/IMvision12/KerasFormers/releases/download/deberta_v3"
DEBERTA_V3_WEIGHT_CONFIG = {
    name: {
        "model": name,
        "url": f"{_REL}/{name}.weights.h5",
        "mlm_url": f"{_REL}/{name}_mlm.weights.h5",
    }
    for name in DEBERTA_V3_MODEL_CONFIG
}

# DeBERTa-v3's SentencePiece tokenizer is shared across all sizes.
DEBERTA_V3_VOCAB_URL = f"{_REL}/deberta_v3_spm.model"
