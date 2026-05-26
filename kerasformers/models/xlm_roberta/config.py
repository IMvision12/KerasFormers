XLM_ROBERTA_MODEL_CONFIG = {
    "xlm_roberta_base": {
        "vocab_size": 250002,
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "max_position_embeddings": 514,
        "type_vocab_size": 1,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-5,
        "pad_token_id": 1,
    },
    "xlm_roberta_large": {
        "vocab_size": 250002,
        "embed_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "mlp_dim": 4096,
        "max_position_embeddings": 514,
        "type_vocab_size": 1,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-5,
        "pad_token_id": 1,
    },
}

XLM_ROBERTA_WEIGHT_CONFIG = {
    "xlm_roberta_base": {
        "model": "xlm_roberta_base",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/xlm_roberta_base.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/xlm_roberta_base_mlm.weights.json",
    },
    "xlm_roberta_large": {
        "model": "xlm_roberta_large",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/xlm_roberta_large.weights.json",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/xlm_roberta_large_mlm.weights.json",
    },
}

XLM_ROBERTA_VOCAB_CONFIG = {
    "xlm_roberta_base": {
        "vocab_url": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/sentencepiece.bpe.model",
    },
    "xlm_roberta_large": {
        "vocab_url": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/sentencepiece.bpe.model",
    },
}
