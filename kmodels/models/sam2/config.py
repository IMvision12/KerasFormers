SAM2_CONFIG = {
    "sam2_hiera_tiny": {
        "hidden_size": 96,
        "blocks_per_stage": [1, 2, 7, 2],
        "embed_dim_per_stage": [96, 192, 384, 768],
        "num_attention_heads_per_stage": [1, 2, 4, 8],
        "window_size_per_stage": [8, 4, 14, 7],
        "global_attention_blocks": [5, 7, 9],
        "backbone_channel_list": [768, 384, 192, 96],
    },
    "sam2_hiera_small": {
        "hidden_size": 96,
        "blocks_per_stage": [1, 2, 11, 2],
        "embed_dim_per_stage": [96, 192, 384, 768],
        "num_attention_heads_per_stage": [1, 2, 4, 8],
        "window_size_per_stage": [8, 4, 14, 7],
        "global_attention_blocks": [7, 10, 13],
        "backbone_channel_list": [768, 384, 192, 96],
    },
    "sam2_hiera_base_plus": {
        "hidden_size": 112,
        "blocks_per_stage": [2, 3, 16, 3],
        "embed_dim_per_stage": [112, 224, 448, 896],
        "num_attention_heads_per_stage": [2, 4, 8, 16],
        "window_size_per_stage": [8, 4, 14, 7],
        "global_attention_blocks": [12, 16, 20],
        "backbone_channel_list": [896, 448, 224, 112],
        "window_pos_embed_bg_size": [14, 14],
    },
    "sam2_hiera_large": {
        "hidden_size": 144,
        "blocks_per_stage": [2, 6, 36, 4],
        "embed_dim_per_stage": [144, 288, 576, 1152],
        "num_attention_heads_per_stage": [2, 4, 8, 16],
        "window_size_per_stage": [8, 4, 16, 8],
        "global_attention_blocks": [23, 33, 43],
        "backbone_channel_list": [1152, 576, 288, 144],
    },
}

SAM2_WEIGHTS = {
    "sam2_hiera_tiny": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/sam2/sam2_hiera_tiny.weights.h5",
    },
    "sam2_hiera_small": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/sam2/sam2_hiera_small.weights.h5",
    },
    "sam2_hiera_base_plus": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/sam2/sam2_hiera_base_plus.weights.h5",
    },
    "sam2_hiera_large": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/sam2/sam2_hiera_large.weights.h5",
    },
}
