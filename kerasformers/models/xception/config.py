XCEPTION_MODEL_CONFIG = {
    "xception41": {
        "config": "41",
        "preact": False,
        "bn_epsilon": 1e-3,
        "input_image_shape": 299,
        "num_classes": 1000,
    },
    "xception41p": {
        "config": "41p",
        "preact": True,
        "bn_epsilon": 1e-5,
        "input_image_shape": 299,
        "num_classes": 1000,
    },
    "xception65": {
        "config": "65",
        "preact": False,
        "bn_epsilon": 1e-3,
        "input_image_shape": 299,
        "num_classes": 1000,
    },
    "xception65p": {
        "config": "65p",
        "preact": True,
        "bn_epsilon": 1e-3,
        "input_image_shape": 299,
        "num_classes": 1000,
    },
    "xception71": {
        "config": "71",
        "preact": False,
        "bn_epsilon": 1e-3,
        "input_image_shape": 299,
        "num_classes": 1000,
    },
}


XCEPTION_WEIGHT_CONFIG = {
    "xception41_tf_in1k": {
        "model": "xception41",
        "timm_id": "xception41.tf_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception41_tf_in1k.weights.h5",
    },
    "xception41p_ra3_in1k": {
        "model": "xception41p",
        "timm_id": "xception41p.ra3_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception41p_ra3_in1k.weights.h5",
    },
    "xception65_ra3_in1k": {
        "model": "xception65",
        "timm_id": "xception65.ra3_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception65_ra3_in1k.weights.h5",
    },
    "xception65_tf_in1k": {
        "model": "xception65",
        "timm_id": "xception65.tf_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception65_tf_in1k.weights.h5",
    },
    "xception65p_ra3_in1k": {
        "model": "xception65p",
        "timm_id": "xception65p.ra3_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception65p_ra3_in1k.weights.h5",
    },
    "xception71_tf_in1k": {
        "model": "xception71",
        "timm_id": "xception71.tf_in1k",
        "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception71_tf_in1k.weights.h5",
    },
}
