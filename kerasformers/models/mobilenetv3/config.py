MOBILENETV3_MODEL_CONFIG = {
    "MobileNetV3Small050": {
        "width_multiplier": 0.5,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": False,
    },
    "MobileNetV3Small075": {
        "width_multiplier": 0.75,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": False,
    },
    "MobileNetV3Small100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "small",
        "minimal": False,
    },
    "MobileNetV3Large100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": False,
    },
    "MobileNetV3Large150d": {
        "width_multiplier": 1.5,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": False,
        "block_count_multiplier": 1.2,
        "head_count_multiplier": 2,
    },
    "MobileNetV3Rw": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "config": "large",
        "minimal": False,
        "first_block_noskip": True,
        "se_round_divisor": None,
        "se_use_block_act": True,
        "bn_epsilon": 1e-3,
        "head_use_bias": False,
    },
}


MOBILENETV3_WEIGHTS_CONFIG = {
    "MobileNetV3Small050": {
        "lamb_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_small_050_lamb_in1k.weights.h5"
        },
    },
    "MobileNetV3Small075": {
        "lamb_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_small_075_lamb_in1k.weights.h5"
        },
    },
    "MobileNetV3Small100": {
        "lamb_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_small_100_lamb_in1k.weights.h5"
        },
    },
    "MobileNetV3Large100": {
        "miil_in21k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_large_100_miil_in21k.weights.h5"
        },
        "miil_in21k_ft_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_large_100_miil_in21k_ft_in1k.weights.h5"
        },
        "ra_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_large_100_ra_in1k.weights.h5"
        },
        "ra4_e3600_r224_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_large_100_ra4_e3600_r224_in1k.weights.h5"
        },
    },
    "MobileNetV3Large150d": {
        "ra4_e3600_r256_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_large_150d_ra4_e3600_r256_in1k.weights.h5"
        },
    },
    "MobileNetV3Rw": {
        "rmsp_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/mobilenetv3_rw_rmsp_in1k.weights.h5"
        },
    },
}
