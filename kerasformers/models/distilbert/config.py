DISTILBERT_MODEL_CONFIG = {
    "distilbert_base_uncased": {
        "vocab_size": 30522,
        "embed_dim": 768,
        "num_layers": 6,
        "num_heads": 12,
        "mlp_dim": 3072,
        "max_position_embeddings": 512,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
    },
    "distilbert_base_cased": {
        "vocab_size": 28996,
        "embed_dim": 768,
        "num_layers": 6,
        "num_heads": 12,
        "mlp_dim": 3072,
        "max_position_embeddings": 512,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
    },
    "distilbert_base_multilingual_cased": {
        "vocab_size": 119547,
        "embed_dim": 768,
        "num_layers": 6,
        "num_heads": 12,
        "mlp_dim": 3072,
        "max_position_embeddings": 512,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
    },
}

DISTILBERT_WEIGHT_CONFIG = {
    "distilbert_base_uncased": {
        "model": "distilbert_base_uncased",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/distilbert/distilbert_base_uncased.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/distilbert/distilbert_base_uncased_mlm.weights.h5",
    },
    "distilbert_base_cased": {
        "model": "distilbert_base_cased",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/distilbert/distilbert_base_cased.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/distilbert/distilbert_base_cased_mlm.weights.h5",
    },
    "distilbert_base_multilingual_cased": {
        "model": "distilbert_base_multilingual_cased",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/distilbert/distilbert_base_multilingual_cased.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/distilbert/distilbert_base_multilingual_cased_mlm.weights.h5",
    },
}

DISTILBERT_VOCAB_CONFIG = {
    "distilbert_base_uncased": {
        "vocab_url": "https://github.com/IMvision12/KerasFormers/releases/download/distilbert/vocab.txt",
        "do_lower_case": True,
    },
    "distilbert_base_cased": {
        "vocab_url": "https://github.com/IMvision12/KerasFormers/releases/download/distilbert/vocab_cased.txt",
        "do_lower_case": False,
    },
    "distilbert_base_multilingual_cased": {
        "vocab_url": "https://github.com/IMvision12/KerasFormers/releases/download/distilbert/vocab_multilingual_cased.txt",
        "do_lower_case": False,
    },
}
