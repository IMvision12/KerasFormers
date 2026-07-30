SAM_CONFIG = {
    "sam_vit_base": {
        "vision_hidden_size": 768,
        "vision_num_hidden_layers": 12,
        "vision_num_attention_heads": 12,
        "vision_mlp_dim": 3072,
        "vision_global_attn_indexes": [2, 5, 8, 11],
        "image_size": 1024,
    },
    "sam_vit_large": {
        "vision_hidden_size": 1024,
        "vision_num_hidden_layers": 24,
        "vision_num_attention_heads": 16,
        "vision_mlp_dim": 4096,
        "vision_global_attn_indexes": [5, 11, 17, 23],
        "image_size": 1024,
    },
    "sam_vit_huge": {
        "vision_hidden_size": 1280,
        "vision_num_hidden_layers": 32,
        "vision_num_attention_heads": 16,
        "vision_mlp_dim": 5120,
        "vision_global_attn_indexes": [7, 15, 23, 31],
        "image_size": 1024,
    },
}

SAM_WEIGHTS_URLS = {
    "sam_vit_base": {
        "url": "https://huggingface.co/kerasformers/sam_vit_base",
    },
    "sam_vit_large": {
        "url": "https://huggingface.co/kerasformers/sam_vit_large",
    },
    "sam_vit_huge": {
        "url": "https://huggingface.co/kerasformers/sam_vit_huge",
    },
}
