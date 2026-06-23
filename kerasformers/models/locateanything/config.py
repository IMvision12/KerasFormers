LOCATEANYTHING_CONFIG = {
    "locateanything_3b": {
        "vocab_size": 152681,
        "embed_dim": 2048,
        "mlp_dim": 11008,
        "num_layers": 36,
        "num_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "tie_embeddings": True,
        "vision_embed_dim": 1152,
        "vision_depth": 27,
        "vision_num_heads": 16,
        "vision_mlp_dim": 4304,
        "vision_patch_size": 14,
        "vision_init_pos_h": 64,
        "vision_init_pos_w": 64,
        "merge_kernel": (2, 2),
        "vision_rope_theta": 10000.0,
        "image_token_index": 151665,
        "block_size": 6,
    },
}

LOCATEANYTHING_WEIGHTS_URLS = {
    "locateanything_3b": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/locateanything/locateanything_3b.weights.json"
    },
}

LOCATEANYTHING_TOKENIZER_URLS = {
    "locateanything_3b": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/locateanything/locateanything_3b_tokenizer.json"
    },
}
