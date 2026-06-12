SWIN_TINY = {
    "backbone_embed_dim": 96,
    "backbone_depths": (2, 2, 6, 2),
    "backbone_num_heads": (3, 6, 12, 24),
    "backbone_window_size": 7,
}

SWIN_LARGE = {
    "backbone_embed_dim": 192,
    "backbone_depths": (2, 2, 18, 2),
    "backbone_num_heads": (6, 12, 24, 48),
    "backbone_window_size": 12,
}

COMMON = {
    "hidden_dim": 256,
    "mask_feature_size": 256,
    "encoder_num_layers": 6,
    "encoder_ffn_dim": 1024,
    "decoder_num_layers": 9,
    "decoder_ffn_dim": 2048,
    "query_dec_layers": 2,
    "num_heads": 8,
    "task_seq_len": 77,
}

ONEFORMER_CONFIG = {
    "oneformer_ade20k_swin_tiny": {
        **SWIN_TINY,
        **COMMON,
        "num_queries": 150,
        "num_classes": 150,
        "image_size": 512,
    },
    "oneformer_ade20k_swin_large": {
        **SWIN_LARGE,
        **COMMON,
        "num_queries": 150,
        "num_classes": 150,
        "image_size": 640,
    },
    "oneformer_coco_swin_large": {
        **SWIN_LARGE,
        **COMMON,
        "num_queries": 150,
        "num_classes": 133,
        "image_size": 800,
    },
    "oneformer_cityscapes_swin_large": {
        **SWIN_LARGE,
        **COMMON,
        "num_queries": 150,
        "num_classes": 19,
        "image_size": 512,
    },
}

ONEFORMER_WEIGHTS_URLS = {
    "oneformer_ade20k_swin_tiny": {
        "hf_id": "shi-labs/oneformer_ade20k_swin_tiny",
        "gated": False,
    },
    "oneformer_ade20k_swin_large": {
        "hf_id": "shi-labs/oneformer_ade20k_swin_large",
        "gated": False,
    },
    "oneformer_coco_swin_large": {
        "hf_id": "shi-labs/oneformer_coco_swin_large",
        "gated": False,
    },
    "oneformer_cityscapes_swin_large": {
        "hf_id": "shi-labs/oneformer_cityscapes_swin_large",
        "gated": False,
    },
}
