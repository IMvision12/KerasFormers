"""EfficientFormer variant registry (timm-ported)."""

_L1 = {
    "depths": [3, 2, 6, 4],
    "embed_dims": [48, 96, 224, 448],
    "num_vit": 1,
}
_L3 = {
    "depths": [4, 4, 12, 6],
    "embed_dims": [64, 128, 320, 512],
    "num_vit": 4,
}
_L7 = {
    "depths": [6, 6, 18, 8],
    "embed_dims": [96, 192, 384, 768],
    "num_vit": 8,
}


def _v(arch, timm_id, image_size, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


EFFICIENTFORMER_CONFIG = {
    "efficientformer_l1_snap_dist_in1k": _v(
        _L1, "efficientformer_l1.snap_dist_in1k", 224
    ),
    "efficientformer_l3_snap_dist_in1k": _v(
        _L3, "efficientformer_l3.snap_dist_in1k", 224
    ),
    "efficientformer_l7_snap_dist_in1k": _v(
        _L7, "efficientformer_l7.snap_dist_in1k", 224
    ),
}

_BASE_URL = (
    "https://github.com/IMvision12/keras-models/releases/download/EfficientFormer"
)
EFFICIENTFORMER_WEIGHTS = {
    "efficientformer_l1_snap_dist_in1k": {
        "url": f"{_BASE_URL}/efficientformer_l1_snap_dist_in1k.weights.h5",
    },
    "efficientformer_l3_snap_dist_in1k": {
        "url": f"{_BASE_URL}/efficientformer_l3_snap_dist_in1k.weights.h5",
    },
    "efficientformer_l7_snap_dist_in1k": {
        "url": f"{_BASE_URL}/efficientformer_l7_snap_dist_in1k.weights.h5",
    },
}
