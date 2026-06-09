GPT2_CONFIG = {
    "gpt2": {
        "vocab_size": 50257,
        "embed_dim": 768,
        "mlp_dim": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
    "gpt2_medium": {
        "vocab_size": 50257,
        "embed_dim": 1024,
        "mlp_dim": 4096,
        "num_layers": 24,
        "num_heads": 16,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
    "gpt2_large": {
        "vocab_size": 50257,
        "embed_dim": 1280,
        "mlp_dim": 5120,
        "num_layers": 36,
        "num_heads": 20,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
    "gpt2_xl": {
        "vocab_size": 50257,
        "embed_dim": 1600,
        "mlp_dim": 6400,
        "num_layers": 48,
        "num_heads": 25,
        "max_position_embeddings": 1024,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
    },
}

GPT2_WEIGHTS_URLS = {
    "gpt2": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/gpt/gpt2.weights.h5"
    },
    "gpt2_medium": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/gpt/gpt2_medium.weights.h5"
    },
    "gpt2_large": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/gpt/gpt2_large.weights.json"
    },
    "gpt2_xl": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/gpt/gpt2_xl.weights.json"
    },
}

GPT2_TOKENIZER_URLS = {
    "gpt2": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/gpt/gpt2_tokenizer.json"
    },
    "gpt2_medium": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/gpt/gpt2_medium_tokenizer.json"
    },
    "gpt2_large": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/gpt/gpt2_large_tokenizer.json"
    },
    "gpt2_xl": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/gpt/gpt2_xl_tokenizer.json"
    },
}
