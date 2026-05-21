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
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RF-DeTR/rf_detr_nano.weights.h5",
    },
    "rfdetr-small": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RF-DeTR/rf_detr_small.weights.h5",
    },
    "rfdetr-medium": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RF-DeTR/rf_detr_medium.weights.h5",
    },
    "rfdetr-base": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RF-DeTR/rf_detr_base.weights.h5",
    },
    "rfdetr-large": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/RF-DeTR/rf_detr_large.weights.h5",
    },
}
