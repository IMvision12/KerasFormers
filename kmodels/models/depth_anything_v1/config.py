DA_V1_CONFIG = {
    "depth_anything_small": {
        "backbone_dim": 384,
        "backbone_depth": 12,
        "backbone_num_heads": 6,
        "out_indices": [9, 10, 11, 12],
        "neck_hidden_sizes": [48, 96, 192, 384],
        "fusion_hidden_size": 64,
        "reassemble_factors": [4, 2, 1, 0.5],
    },
    "depth_anything_base": {
        "backbone_dim": 768,
        "backbone_depth": 12,
        "backbone_num_heads": 12,
        "out_indices": [9, 10, 11, 12],
        "neck_hidden_sizes": [96, 192, 384, 768],
        "fusion_hidden_size": 128,
        "reassemble_factors": [4, 2, 1, 0.5],
    },
    "depth_anything_large": {
        "backbone_dim": 1024,
        "backbone_depth": 24,
        "backbone_num_heads": 16,
        "out_indices": [21, 22, 23, 24],
        "neck_hidden_sizes": [256, 512, 1024, 1024],
        "fusion_hidden_size": 256,
        "reassemble_factors": [4, 2, 1, 0.5],
    },
}

_V1_BASE_URL = (
    "https://github.com/IMvision12/keras-models/releases/download/depth-anything-v1"
)

DA_V1_WEIGHTS = {
    "depth_anything_small": {"url": f"{_V1_BASE_URL}/depth_anything_small.weights.h5"},
    "depth_anything_base": {"url": f"{_V1_BASE_URL}/depth_anything_base.weights.h5"},
    "depth_anything_large": {"url": f"{_V1_BASE_URL}/depth_anything_large.weights.h5"},
}
