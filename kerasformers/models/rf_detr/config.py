RF_DETR_DETECT_CONFIG = {
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

RF_DETR_DETECT_WEIGHTS_URLS = {
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

RF_DETR_SEGMENT_WEIGHTS_URLS = {
    "rfdetr-seg-preview": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_seg_preview.weights.h5",
    },
    "rfdetr-seg-nano": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_seg_nano.weights.h5",
    },
    "rfdetr-seg-small": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_seg_small.weights.h5",
    },
    "rfdetr-seg-medium": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_seg_medium.weights.h5",
    },
    "rfdetr-seg-large": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_seg_large.weights.h5",
    },
    "rfdetr-seg-xlarge": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_seg_xlarge.weights.h5",
    },
    "rfdetr-seg-xxlarge": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/rf-detr/rf_detr_seg_xxlarge.weights.h5",
    },
}
