MOONSHINE_CONFIG = {
    "moonshine_tiny": {
        "hidden_dim": 288,
        "encoder_num_layers": 6,
        "decoder_num_layers": 6,
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 8,
        "encoder_ffn_dim": 1152,
        "decoder_ffn_dim": 1152,
        "vocab_size": 32768,
        "max_position_embeddings": 194,
        "partial_rotary_factor": 0.9,
        "rope_theta": 10000.0,
        "encoder_activation": "gelu",
        "decoder_activation": "silu",
    },
    "moonshine_base": {
        "hidden_dim": 416,
        "encoder_num_layers": 8,
        "decoder_num_layers": 8,
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 8,
        "encoder_ffn_dim": 1664,
        "decoder_ffn_dim": 1664,
        "vocab_size": 32768,
        "max_position_embeddings": 194,
        "partial_rotary_factor": 0.62,
        "rope_theta": 10000.0,
        "encoder_activation": "gelu",
        "decoder_activation": "silu",
    },
}


MOONSHINE_WEIGHTS_URLS = {
    "moonshine_tiny": {"url": "https://huggingface.co/kerasformers/moonshine_tiny"},
    "moonshine_base": {"url": "https://huggingface.co/kerasformers/moonshine_base"},
}


MOONSHINE_TOKENIZER_URLS = {
    "moonshine_tiny": {
        "tokenizer_json": "https://huggingface.co/kerasformers/moonshine_tiny/resolve/main/tokenizer.json"
    },
    "moonshine_base": {
        "tokenizer_json": "https://huggingface.co/kerasformers/moonshine_base/resolve/main/tokenizer.json"
    },
}
