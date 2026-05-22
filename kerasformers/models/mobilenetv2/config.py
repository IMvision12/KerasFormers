MOBILENETV2_MODEL_CONFIG = {
    "mobilenetv2_050": {
        "width_multiplier": 0.5,
        "depth_multiplier": 1.0,
        "fix_channels": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv2_100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "fix_channels": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv2_110d": {
        "width_multiplier": 1.1,
        "depth_multiplier": 1.2,
        "fix_channels": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv2_120d": {
        "width_multiplier": 1.2,
        "depth_multiplier": 1.4,
        "fix_channels": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv2_140": {
        "width_multiplier": 1.4,
        "depth_multiplier": 1.0,
        "fix_channels": False,
        "image_size": 224,
        "num_classes": 1000,
    },
}

MOBILENETV2_WEIGHT_CONFIG = {
    "mobilenetv2_050_lamb_in1k": {
        "model": "mobilenetv2_050",
        "timm_id": "mobilenetv2_050.lamb_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/mobilenetv2_050_lamb_in1k.weights.h5",
    },
    "mobilenetv2_100_ra_in1k": {
        "model": "mobilenetv2_100",
        "timm_id": "mobilenetv2_100.ra_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/mobilenetv2_100_ra_in1k.weights.h5",
    },
    "mobilenetv2_110d_ra_in1k": {
        "model": "mobilenetv2_110d",
        "timm_id": "mobilenetv2_110d.ra_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/mobilenetv2_110d_ra_in1k.weights.h5",
    },
    "mobilenetv2_120d_ra_in1k": {
        "model": "mobilenetv2_120d",
        "timm_id": "mobilenetv2_120d.ra_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/mobilenetv2_120d_ra_in1k.weights.h5",
    },
    "mobilenetv2_140_ra_in1k": {
        "model": "mobilenetv2_140",
        "timm_id": "mobilenetv2_140.ra_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/mobilenetv2_140_ra_in1k.weights.h5",
    },
}
