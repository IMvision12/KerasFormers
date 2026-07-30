CLIP_CONFIG = {
    "clip_vit_base_16": {
        "embed_dim": 512,
        "image_size": 224,
        "vision_num_layers": 12,
        "vision_hidden_dim": 768,
        "vision_patch_size": 16,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 512,
        "text_num_heads": 8,
        "text_num_layers": 12,
        "vision_mlp_ratio": 4.0,
        "text_mlp_ratio": 4.0,
        "hidden_act": "quick_gelu",
    },
    "clip_vit_base_32": {
        "embed_dim": 512,
        "image_size": 224,
        "vision_num_layers": 12,
        "vision_hidden_dim": 768,
        "vision_patch_size": 32,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 512,
        "text_num_heads": 8,
        "text_num_layers": 12,
        "vision_mlp_ratio": 4.0,
        "text_mlp_ratio": 4.0,
        "hidden_act": "quick_gelu",
    },
    "clip_vit_large_14": {
        "embed_dim": 768,
        "image_size": 224,
        "vision_num_layers": 24,
        "vision_hidden_dim": 1024,
        "vision_patch_size": 14,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 768,
        "text_num_heads": 12,
        "text_num_layers": 12,
        "vision_mlp_ratio": 4.0,
        "text_mlp_ratio": 4.0,
        "hidden_act": "quick_gelu",
    },
    "clip_vit_large_14_336": {
        "embed_dim": 768,
        "image_size": 336,
        "vision_num_layers": 24,
        "vision_hidden_dim": 1024,
        "vision_patch_size": 14,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 768,
        "text_num_heads": 12,
        "text_num_layers": 12,
        "vision_mlp_ratio": 4.0,
        "text_mlp_ratio": 4.0,
        "hidden_act": "quick_gelu",
    },
    "clip_vit_g_14": {
        "embed_dim": 1024,
        "image_size": 224,
        "vision_num_layers": 40,
        "vision_hidden_dim": 1408,
        "vision_patch_size": 14,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 1024,
        "text_num_heads": 16,
        "text_num_layers": 24,
        "vision_mlp_ratio": 6144 / 1408,
        "text_mlp_ratio": 4096 / 1024,
        "hidden_act": "gelu",
    },
    "clip_vit_bigg_14": {
        "embed_dim": 1280,
        "image_size": 224,
        "vision_num_layers": 48,
        "vision_hidden_dim": 1664,
        "vision_patch_size": 14,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 1280,
        "text_num_heads": 20,
        "text_num_layers": 32,
        "vision_mlp_ratio": 8192 / 1664,
        "text_mlp_ratio": 5120 / 1280,
        "hidden_act": "gelu",
    },
}

CLIP_WEIGHTS_URLS = {
    "clip_vit_base_16": {"url": "https://huggingface.co/kerasformers/clip_vit_base_16"},
    "clip_vit_base_32": {"url": "https://huggingface.co/kerasformers/clip_vit_base_32"},
    "clip_vit_large_14": {
        "url": "https://huggingface.co/kerasformers/clip_vit_large_14"
    },
    "clip_vit_large_14_336": {
        "url": "https://huggingface.co/kerasformers/clip_vit_large_14_336"
    },
    "clip_vit_g_14": {"url": "https://huggingface.co/kerasformers/clip_vit_g_14"},
    "clip_vit_bigg_14": {"url": "https://huggingface.co/kerasformers/clip_vit_bigg_14"},
}


CLIP_TOKENIZER_URLS = {
    "clip_vit_base_16": {
        "tokenizer_json": "https://huggingface.co/kerasformers/clip_vit_base_16/resolve/main/tokenizer.json"
    },
    "clip_vit_base_32": {
        "tokenizer_json": "https://huggingface.co/kerasformers/clip_vit_base_32/resolve/main/tokenizer.json"
    },
    "clip_vit_large_14": {
        "tokenizer_json": "https://huggingface.co/kerasformers/clip_vit_large_14/resolve/main/tokenizer.json"
    },
    "clip_vit_large_14_336": {
        "tokenizer_json": "https://huggingface.co/kerasformers/clip_vit_large_14_336/resolve/main/tokenizer.json"
    },
    "clip_vit_g_14": {
        "tokenizer_json": "https://huggingface.co/kerasformers/clip_vit_g_14/resolve/main/tokenizer.json"
    },
    "clip_vit_bigg_14": {
        "tokenizer_json": "https://huggingface.co/kerasformers/clip_vit_bigg_14/resolve/main/tokenizer.json"
    },
}
