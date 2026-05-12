_V2_SMALL_BASE = {
    "backbone_dim": 384,
    "backbone_depth": 12,
    "backbone_num_heads": 6,
    "out_indices": [3, 6, 9, 12],
    "neck_hidden_sizes": [48, 96, 192, 384],
    "fusion_hidden_size": 64,
    "reassemble_factors": [4, 2, 1, 0.5],
}

_V2_BASE_BASE = {
    "backbone_dim": 768,
    "backbone_depth": 12,
    "backbone_num_heads": 12,
    "out_indices": [3, 6, 9, 12],
    "neck_hidden_sizes": [96, 192, 384, 768],
    "fusion_hidden_size": 128,
    "reassemble_factors": [4, 2, 1, 0.5],
}

_V2_LARGE_BASE = {
    "backbone_dim": 1024,
    "backbone_depth": 24,
    "backbone_num_heads": 16,
    "out_indices": [5, 12, 18, 24],
    "neck_hidden_sizes": [256, 512, 1024, 1024],
    "fusion_hidden_size": 256,
    "reassemble_factors": [4, 2, 1, 0.5],
}

DA_V2_CONFIG = {
    "depth_anything_v2_small": {**_V2_SMALL_BASE},
    "depth_anything_v2_base": {**_V2_BASE_BASE},
    "depth_anything_v2_large": {**_V2_LARGE_BASE},
    "depth_anything_v2_metric_indoor_small": {
        **_V2_SMALL_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 20.0,
    },
    "depth_anything_v2_metric_indoor_base": {
        **_V2_BASE_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 20.0,
    },
    "depth_anything_v2_metric_indoor_large": {
        **_V2_LARGE_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 20.0,
    },
    "depth_anything_v2_metric_outdoor_small": {
        **_V2_SMALL_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 80.0,
    },
    "depth_anything_v2_metric_outdoor_base": {
        **_V2_BASE_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 80.0,
    },
    "depth_anything_v2_metric_outdoor_large": {
        **_V2_LARGE_BASE,
        "depth_estimation_type": "metric",
        "max_depth": 80.0,
    },
}

_BASE_URL = (
    "https://github.com/IMvision12/keras-models/releases/download/depth-anything-v2"
)

DA_V2_WEIGHTS = {
    name: {"url": f"{_BASE_URL}/{name}.weights.h5"} for name in DA_V2_CONFIG
}
