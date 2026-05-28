RF_DETR_CONFIG = {
    "rfdetr-nano": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 16,
        "num_windows": 2,
        "positional_encoding_size": 24,
        "resolution": 384,
        "dec_layers": 2,
    },
    "rfdetr-small": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 16,
        "num_windows": 2,
        "positional_encoding_size": 32,
        "resolution": 512,
        "dec_layers": 3,
    },
    "rfdetr-medium": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 16,
        "num_windows": 2,
        "positional_encoding_size": 36,
        "resolution": 576,
        "dec_layers": 4,
    },
    "rfdetr-base": {
        "out_feature_indexes": [2, 5, 8, 11],
        "patch_size": 14,
        "num_windows": 4,
        "positional_encoding_size": 37,
        "resolution": 560,
        "dec_layers": 3,
    },
    "rfdetr-large": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 16,
        "num_windows": 2,
        "positional_encoding_size": 44,
        "resolution": 704,
        "dec_layers": 4,
    },
}

RF_DETR_WEIGHTS = {
    "rfdetr-nano": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_nano.weights.h5",
    },
    "rfdetr-small": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_small.weights.h5",
    },
    "rfdetr-medium": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_medium.weights.h5",
    },
    "rfdetr-base": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_base.weights.h5",
    },
    "rfdetr-large": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_large.weights.h5",
    },
}

# Instance-segmentation variants. Same DINOv2 backbone + deformable decoder as the
# detection variants (all patch_size 12), plus a mask head. `num_queries` and
# `dec_layers` vary per size; `positional_encoding_size = resolution // patch_size`.
RF_DETR_SEGMENT_CONFIG = {
    "rfdetr-seg-preview": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 36,
        "resolution": 432,
        "dec_layers": 4,
        "num_queries": 200,
    },
    "rfdetr-seg-nano": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 1,
        "positional_encoding_size": 26,
        "resolution": 312,
        "dec_layers": 4,
        "num_queries": 100,
    },
    "rfdetr-seg-small": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 32,
        "resolution": 384,
        "dec_layers": 4,
        "num_queries": 100,
    },
    "rfdetr-seg-medium": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 36,
        "resolution": 432,
        "dec_layers": 5,
        "num_queries": 200,
    },
    "rfdetr-seg-large": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 42,
        "resolution": 504,
        "dec_layers": 5,
        "num_queries": 300,
    },
    "rfdetr-seg-xlarge": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 52,
        "resolution": 624,
        "dec_layers": 6,
        "num_queries": 300,
    },
    "rfdetr-seg-xxlarge": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 64,
        "resolution": 768,
        "dec_layers": 6,
        "num_queries": 300,
    },
}

_SEG_REL = "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr-seg"
RF_DETR_SEGMENT_WEIGHTS = {
    name: {"url": f"{_SEG_REL}/rf_detr_{name.replace('-', '_')}.weights.h5"}
    for name in RF_DETR_SEGMENT_CONFIG
}
