BERT_MODEL_CONFIG = {
    "bert_base_uncased": {
        "vocab_size": 30522,
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
    },
    "bert_large_uncased": {
        "vocab_size": 30522,
        "embed_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "mlp_dim": 4096,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
    },
    "bert_base_cased": {
        "vocab_size": 28996,
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
    },
    "bert_large_cased": {
        "vocab_size": 28996,
        "embed_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "mlp_dim": 4096,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
    },
}

BERT_WEIGHT_CONFIG = {
    "bert_base_uncased": {
        "model": "bert_base_uncased",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/bert_base_uncased.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/bert_base_uncased_mlm.weights.h5",
    },
    "bert_large_uncased": {
        "model": "bert_large_uncased",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/bert_large_uncased.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/bert_large_uncased_mlm.weights.h5",
    },
    "bert_base_cased": {
        "model": "bert_base_cased",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/bert_base_cased.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/bert_base_cased_mlm.weights.h5",
    },
    "bert_large_cased": {
        "model": "bert_large_cased",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/bert_large_cased.weights.h5",
        "mlm_url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/bert_large_cased_mlm.weights.h5",
    },
}

BERT_VOCAB_CONFIG = {
    "bert_base_uncased": {
        "vocab_url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/vocab.txt",
        "do_lower_case": True,
    },
    "bert_large_uncased": {
        "vocab_url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/vocab.txt",
        "do_lower_case": True,
    },
    "bert_base_cased": {
        "vocab_url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/vocab_cased.txt",
        "do_lower_case": False,
    },
    "bert_large_cased": {
        "vocab_url": "https://github.com/IMvision12/KerasFormers/releases/download/bert/vocab_cased.txt",
        "do_lower_case": False,
    },
}
