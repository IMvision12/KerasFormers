EOMT_CONFIG = {
    "eomt_small_coco_panoptic_640": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "num_blocks": 3,
        "layerscale_value": 1.0,
        "num_queries": 200,
        "num_classes": 133,
        "input_image_shape": 640,
    },
    "eomt_base_coco_panoptic_640": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_blocks": 3,
        "layerscale_value": 1.0,
        "num_queries": 200,
        "num_classes": 133,
        "input_image_shape": 640,
    },
    "eomt_large_coco_panoptic_640": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_blocks": 4,
        "layerscale_value": 1e-5,
        "num_queries": 200,
        "num_classes": 133,
        "input_image_shape": 640,
    },
    "eomt_large_coco_instance_640": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_blocks": 4,
        "layerscale_value": 1e-5,
        "num_queries": 200,
        "num_classes": 80,
        "input_image_shape": 640,
    },
    "eomt_large_ade20k_semantic_512": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_blocks": 4,
        "layerscale_value": 1e-5,
        "num_queries": 100,
        "num_classes": 150,
        "input_image_shape": 512,
    },
}

EOMT_WEIGHTS = {
    "eomt_small_coco_panoptic_640": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/eomt/eomt_small_coco_panoptic_640.weights.h5",
    },
    "eomt_base_coco_panoptic_640": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/eomt/eomt_base_coco_panoptic_640.weights.h5",
    },
    "eomt_large_coco_panoptic_640": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/eomt/eomt_large_coco_panoptic_640.weights.h5",
    },
    "eomt_large_coco_instance_640": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/eomt/eomt_large_coco_instance_640.weights.h5",
    },
    "eomt_large_ade20k_semantic_512": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/eomt/eomt_large_ade20k_semantic_512.weights.h5",
    },
}
