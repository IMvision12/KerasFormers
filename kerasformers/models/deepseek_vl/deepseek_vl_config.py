DEEPSEEK_VL_CONFIG = {
    "deepseek_vl_1.3b_chat": {
        "vocab_size": 102400,
        "embed_dim": 2048,
        "mlp_dim": 5632,
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 16,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_embeddings": False,
        "vision_embed_dim": 1024,
        "vision_mlp_dim": 4096,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "image_size": 384,
        "patch_size": 16,
        "vision_norm_eps": 1e-6,
        "image_token_id": 100015,
    },
    "deepseek_vl_1.3b_base": {
        "vocab_size": 102400,
        "embed_dim": 2048,
        "mlp_dim": 5632,
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 16,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_embeddings": False,
        "vision_embed_dim": 1024,
        "vision_mlp_dim": 4096,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "image_size": 384,
        "patch_size": 16,
        "vision_norm_eps": 1e-6,
        "image_token_id": 100015,
    },
}

DEEPSEEK_VL_WEIGHTS_URLS = {
    "deepseek_vl_1.3b_chat": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/deepseek_vl/deepseek_vl_1.3b_chat.weights.json"
    },
    "deepseek_vl_1.3b_base": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/deepseek_vl/deepseek_vl_1.3b_base.weights.json"
    },
}

DEEPSEEK_VL_TOKENIZER_URLS = {
    "deepseek_vl_1.3b_chat": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/deepseek_vl/deepseek_vl_1.3b_chat_tokenizer.json"
    },
    "deepseek_vl_1.3b_base": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/deepseek_vl/deepseek_vl_1.3b_base_tokenizer.json"
    },
}
