ONEFORMER_CONFIG = {
    "oneformer_ade20k_swin_tiny": {
        "backbone_embed_dim": 96,
        "backbone_depths": (2, 2, 6, 2),
        "backbone_num_heads": (3, 6, 12, 24),
        "backbone_window_size": 7,
        "num_classes": 150,
        "image_size": 512,
    },
    "oneformer_ade20k_swin_large": {
        "backbone_embed_dim": 192,
        "backbone_depths": (2, 2, 18, 2),
        "backbone_num_heads": (6, 12, 24, 48),
        "backbone_window_size": 12,
        "num_classes": 150,
        "image_size": 640,
    },
    "oneformer_coco_swin_large": {
        "backbone_embed_dim": 192,
        "backbone_depths": (2, 2, 18, 2),
        "backbone_num_heads": (6, 12, 24, 48),
        "backbone_window_size": 12,
        "num_classes": 133,
        "image_size": 800,
    },
    "oneformer_cityscapes_swin_large": {
        "backbone_embed_dim": 192,
        "backbone_depths": (2, 2, 18, 2),
        "backbone_num_heads": (6, 12, 24, 48),
        "backbone_window_size": 12,
        "num_classes": 19,
        "image_size": 512,
    },
}

ONEFORMER_WEIGHTS_URLS = {
    "oneformer_ade20k_swin_tiny": {
        "url": "https://huggingface.co/kerasformers/oneformer_ade20k_swin_tiny"
    },
    "oneformer_ade20k_swin_large": {
        "url": "https://huggingface.co/kerasformers/oneformer_ade20k_swin_large"
    },
    "oneformer_coco_swin_large": {
        "url": "https://huggingface.co/kerasformers/oneformer_coco_swin_large"
    },
    "oneformer_cityscapes_swin_large": {
        "url": "https://huggingface.co/kerasformers/oneformer_cityscapes_swin_large"
    },
}

ONEFORMER_TOKENIZER_URLS = {
    "oneformer_ade20k_swin_tiny": {
        "tokenizer_json": "https://huggingface.co/kerasformers/oneformer_ade20k_swin_tiny/resolve/main/tokenizer.json"
    },
    "oneformer_ade20k_swin_large": {
        "tokenizer_json": "https://huggingface.co/kerasformers/oneformer_ade20k_swin_large/resolve/main/tokenizer.json"
    },
    "oneformer_coco_swin_large": {
        "tokenizer_json": "https://huggingface.co/kerasformers/oneformer_coco_swin_large/resolve/main/tokenizer.json"
    },
    "oneformer_cityscapes_swin_large": {
        "tokenizer_json": "https://huggingface.co/kerasformers/oneformer_cityscapes_swin_large/resolve/main/tokenizer.json"
    },
}
