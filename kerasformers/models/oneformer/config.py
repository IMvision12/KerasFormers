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
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/oneformer/oneformer_ade20k_swin_tiny.weights.h5"
    },
    "oneformer_ade20k_swin_large": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/oneformer/oneformer_ade20k_swin_large.weights.h5"
    },
    "oneformer_coco_swin_large": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/oneformer/oneformer_coco_swin_large.weights.h5"
    },
    "oneformer_cityscapes_swin_large": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/oneformer/oneformer_cityscapes_swin_large.weights.h5"
    },
}

ONEFORMER_TOKENIZER_URLS = {
    "oneformer_ade20k_swin_tiny": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/oneformer/oneformer_ade20k_swin_tiny_tokenizer.json"
    },
    "oneformer_ade20k_swin_large": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/oneformer/oneformer_ade20k_swin_large_tokenizer.json"
    },
    "oneformer_coco_swin_large": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/oneformer/oneformer_coco_swin_large_tokenizer.json"
    },
    "oneformer_cityscapes_swin_large": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/oneformer/oneformer_cityscapes_swin_large_tokenizer.json"
    },
}
