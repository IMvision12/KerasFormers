_EOMT_SMALL_BASE = {
    "hidden_size": 384,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "num_blocks": 3,
    "layerscale_value": 1.0,
}

_EOMT_BASE_BASE = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_blocks": 3,
    "layerscale_value": 1.0,
}

_EOMT_LARGE_BASE = {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_blocks": 4,
    "layerscale_value": 1e-5,
}


EOMT_CONFIG = {
    "eomt_small_coco_panoptic_640": {
        **_EOMT_SMALL_BASE,
        "num_queries": 200,
        "num_labels": 133,
        "input_shape": (640, 640, 3),
    },
    "eomt_base_coco_panoptic_640": {
        **_EOMT_BASE_BASE,
        "num_queries": 200,
        "num_labels": 133,
        "input_shape": (640, 640, 3),
    },
    "eomt_large_coco_panoptic_640": {
        **_EOMT_LARGE_BASE,
        "num_queries": 200,
        "num_labels": 133,
        "input_shape": (640, 640, 3),
    },
    "eomt_large_coco_instance_640": {
        **_EOMT_LARGE_BASE,
        "num_queries": 200,
        "num_labels": 80,
        "input_shape": (640, 640, 3),
    },
    "eomt_large_ade20k_semantic_512": {
        **_EOMT_LARGE_BASE,
        "num_queries": 100,
        "num_labels": 150,
        "input_shape": (512, 512, 3),
    },
}

EOMT_WEIGHTS = {
    "eomt_small_coco_panoptic_640": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/eomt/eomt_small_coco_panoptic_640.weights.h5",
    },
    "eomt_base_coco_panoptic_640": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/eomt/eomt_base_coco_panoptic_640.weights.h5",
    },
    "eomt_large_coco_panoptic_640": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/eomt/eomt_large_coco_panoptic_640.weights.h5",
    },
    "eomt_large_coco_instance_640": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/eomt/eomt_large_coco_instance_640.weights.h5",
    },
    "eomt_large_ade20k_semantic_512": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/eomt/eomt_large_ade20k_semantic_512.weights.h5",
    },
}
