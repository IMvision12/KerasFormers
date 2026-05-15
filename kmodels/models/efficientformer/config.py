EFFICIENTFORMER_MODEL_CONFIG = {
    "efficientformer_l1": {
        "depths": [3, 2, 6, 4],
        "embed_dims": [48, 96, 224, 448],
        "num_vit": 1,
        "image_size": 224,
        "num_classes": 1000,
    },
    "efficientformer_l3": {
        "depths": [4, 4, 12, 6],
        "embed_dims": [64, 128, 320, 512],
        "num_vit": 4,
        "image_size": 224,
        "num_classes": 1000,
    },
    "efficientformer_l7": {
        "depths": [6, 6, 18, 8],
        "embed_dims": [96, 192, 384, 768],
        "num_vit": 8,
        "image_size": 224,
        "num_classes": 1000,
    },
}

EFFICIENTFORMER_WEIGHT_CONFIG = {
    "efficientformer_l1_snap_dist_in1k": {
        "model": "efficientformer_l1",
        "timm_id": "efficientformer_l1.snap_dist_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/EfficientFormer/efficientformer_l1_snap_dist_in1k.weights.h5",
    },
    "efficientformer_l3_snap_dist_in1k": {
        "model": "efficientformer_l3",
        "timm_id": "efficientformer_l3.snap_dist_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/EfficientFormer/efficientformer_l3_snap_dist_in1k.weights.h5",
    },
    "efficientformer_l7_snap_dist_in1k": {
        "model": "efficientformer_l7",
        "timm_id": "efficientformer_l7.snap_dist_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/EfficientFormer/efficientformer_l7_snap_dist_in1k.weights.h5",
    },
}
