RESNETV2_MODEL_CONFIG = {
    "resnetv2_50x1_bit_goog_in21k": {
        "block_repeats": [3, 4, 6, 3],
        "width_factor": 1,
        "timm_id": "resnetv2_50x1_bit.goog_in21k",
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_50x1_bit_goog_in21k_ft_in1k": {
        "block_repeats": [3, 4, 6, 3],
        "width_factor": 1,
        "timm_id": "resnetv2_50x1_bit.goog_in21k_ft_in1k",
        "image_size": 448,
        "num_classes": 1000,
    },
    "resnetv2_50x3_bit_goog_in21k": {
        "block_repeats": [3, 4, 6, 3],
        "width_factor": 3,
        "timm_id": "resnetv2_50x3_bit.goog_in21k",
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_50x3_bit_goog_in21k_ft_in1k": {
        "block_repeats": [3, 4, 6, 3],
        "width_factor": 3,
        "timm_id": "resnetv2_50x3_bit.goog_in21k_ft_in1k",
        "image_size": 448,
        "num_classes": 1000,
    },
    "resnetv2_101x1_bit_goog_in21k": {
        "block_repeats": [3, 4, 23, 3],
        "width_factor": 1,
        "timm_id": "resnetv2_101x1_bit.goog_in21k",
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_101x1_bit_goog_in21k_ft_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "width_factor": 1,
        "timm_id": "resnetv2_101x1_bit.goog_in21k_ft_in1k",
        "image_size": 448,
        "num_classes": 1000,
    },
    "resnetv2_101x3_bit_goog_in21k": {
        "block_repeats": [3, 4, 23, 3],
        "width_factor": 3,
        "timm_id": "resnetv2_101x3_bit.goog_in21k",
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_101x3_bit_goog_in21k_ft_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "width_factor": 3,
        "timm_id": "resnetv2_101x3_bit.goog_in21k_ft_in1k",
        "image_size": 448,
        "num_classes": 1000,
    },
    "resnetv2_152x2_bit_goog_in21k": {
        "block_repeats": [3, 8, 36, 3],
        "width_factor": 2,
        "timm_id": "resnetv2_152x2_bit.goog_in21k",
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_152x2_bit_goog_in21k_ft_in1k": {
        "block_repeats": [3, 8, 36, 3],
        "width_factor": 2,
        "timm_id": "resnetv2_152x2_bit.goog_in21k_ft_in1k",
        "image_size": 448,
        "num_classes": 1000,
    },
    "resnetv2_152x4_bit_goog_in21k": {
        "block_repeats": [3, 8, 36, 3],
        "width_factor": 4,
        "timm_id": "resnetv2_152x4_bit.goog_in21k",
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_152x4_bit_goog_in21k_ft_in1k": {
        "block_repeats": [3, 8, 36, 3],
        "width_factor": 4,
        "timm_id": "resnetv2_152x4_bit.goog_in21k_ft_in1k",
        "image_size": 480,
        "num_classes": 1000,
    },
}

RESNETV2_WEIGHT_CONFIG = {
    "resnetv2_50x1_bit_goog_in21k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_50x1_bit_goog_in21k.weights.h5",
    },
    "resnetv2_50x1_bit_goog_in21k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_50x1_bit_goog_in21k_ft_in1k.weights.h5",
    },
    "resnetv2_50x3_bit_goog_in21k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_50x3_bit_goog_in21k.weights.h5",
    },
    "resnetv2_50x3_bit_goog_in21k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_50x3_bit_goog_in21k_ft_in1k.weights.h5",
    },
    "resnetv2_101x1_bit_goog_in21k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_101x1_bit_goog_in21k.weights.h5",
    },
    "resnetv2_101x1_bit_goog_in21k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_101x1_bit_goog_in21k_ft_in1k.weights.h5",
    },
    "resnetv2_101x3_bit_goog_in21k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_101x3_bit_goog_in21k.weights.h5",
    },
    "resnetv2_101x3_bit_goog_in21k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_101x3_bit_goog_in21k_ft_in1k.weights.h5",
    },
    "resnetv2_152x2_bit_goog_in21k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_152x2_bit_goog_in21k.weights.h5",
    },
    "resnetv2_152x2_bit_goog_in21k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_152x2_bit_goog_in21k_ft_in1k.weights.h5",
    },
    "resnetv2_152x4_bit_goog_in21k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_152x4_bit_goog_in21k.weights.json",
    },
    "resnetv2_152x4_bit_goog_in21k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnetv2_152x4_bit_goog_in21k_ft_in1k.weights.json",
    },
}
