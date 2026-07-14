ROBERTA_MODEL_CONFIG = {
    "roberta_base": {
        "vocab_size": 50265,
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
    "roberta_large": {
        "vocab_size": 50265,
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

ROBERTA_WEIGHTS_URLS = {
    "roberta_base": {
        "model": "roberta_base",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/roberta_base.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/roberta_base_mlm.weights.h5",
    },
    "roberta_large": {
        "model": "roberta_large",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/roberta_large.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/roberta_large_mlm.weights.h5",
    },
}

ROBERTA_TOKENIZER_URLS = {
    "roberta_base": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/roberta_base_tokenizer.json"
    },
    "roberta_large": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/roberta/roberta_large_tokenizer.json"
    },
}
