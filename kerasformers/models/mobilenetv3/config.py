MOBILENETV3_MODEL_CONFIG = {
    "mobilenetv3_small_050": {
        "width_multiplier": 0.5,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": False,
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "mobilenetv3_small_075": {
        "width_multiplier": 0.75,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": False,
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "mobilenetv3_small_100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": False,
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "mobilenetv3_large_100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": False,
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "mobilenetv3_large_150d": {
        "width_multiplier": 1.5,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": False,
        "block_count_multiplier": 1.2,
        "head_count_multiplier": 2,
        "input_image_shape": 256,
        "num_classes": 1000,
    },
    "mobilenetv3_rw": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": False,
        "first_block_noskip": True,
        "se_round_divisor": None,
        "se_use_block_act": True,
        "bn_epsilon": 1e-3,
        "head_use_bias": False,
        "input_image_shape": 224,
        "num_classes": 1000,
    },
}


MOBILENETV3_WEIGHT_CONFIG = {
    "mobilenetv3_small_050_lamb_in1k": {
        "model": "mobilenetv3_small_050",
        "timm_id": "mobilenetv3_small_050.lamb_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_small_050_lamb_in1k.weights.h5",
    },
    "mobilenetv3_small_075_lamb_in1k": {
        "model": "mobilenetv3_small_075",
        "timm_id": "mobilenetv3_small_075.lamb_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_small_075_lamb_in1k.weights.h5",
    },
    "mobilenetv3_small_100_lamb_in1k": {
        "model": "mobilenetv3_small_100",
        "timm_id": "mobilenetv3_small_100.lamb_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_small_100_lamb_in1k.weights.h5",
    },
    "mobilenetv3_large_100_miil_in21k": {
        "model": "mobilenetv3_large_100",
        "timm_id": "mobilenetv3_large_100.miil_in21k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_large_100_miil_in21k.weights.h5",
    },
    "mobilenetv3_large_100_miil_in21k_ft_in1k": {
        "model": "mobilenetv3_large_100",
        "timm_id": "mobilenetv3_large_100.miil_in21k_ft_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_large_100_miil_in21k_ft_in1k.weights.h5",
    },
    "mobilenetv3_large_100_ra_in1k": {
        "model": "mobilenetv3_large_100",
        "timm_id": "mobilenetv3_large_100.ra_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_large_100_ra_in1k.weights.h5",
    },
    "mobilenetv3_large_100_ra4_e3600_r224_in1k": {
        "model": "mobilenetv3_large_100",
        "timm_id": "mobilenetv3_large_100.ra4_e3600_r224_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_large_100_ra4_e3600_r224_in1k.weights.h5",
    },
    "mobilenetv3_large_150d_ra4_e3600_r256_in1k": {
        "model": "mobilenetv3_large_150d",
        "timm_id": "mobilenetv3_large_150d.ra4_e3600_r256_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_large_150d_ra4_e3600_r256_in1k.weights.h5",
    },
    "mobilenetv3_rw_rmsp_in1k": {
        "model": "mobilenetv3_rw",
        "timm_id": "mobilenetv3_rw.rmsp_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_rw_rmsp_in1k.weights.h5",
    },
}
