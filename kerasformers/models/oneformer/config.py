# Only the per-variant differences live here; everything common to all OneFormer
# checkpoints (hidden_dim, mask_feature_size, encoder/decoder depths and ffn
# dims, query_dec_layers, num_heads, task_seq_len, num_queries) is baked into the
# OneFormerModel.__init__ defaults. The per-variant Swin backbone shape, number
# of classes and input size are kept here.
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
