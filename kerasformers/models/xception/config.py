XCEPTION_MODEL_CONFIG = {
    "Xception41": {
        "config": "41",
        "preact": False,
        "bn_epsilon": 1e-3,
    },
    "Xception41p": {
        "config": "41p",
        "preact": True,
        "bn_epsilon": 1e-5,
    },
    "Xception65": {
        "config": "65",
        "preact": False,
        "bn_epsilon": 1e-3,
    },
    "Xception65p": {
        "config": "65p",
        "preact": True,
        "bn_epsilon": 1e-3,
    },
    "Xception71": {
        "config": "71",
        "preact": False,
        "bn_epsilon": 1e-3,
    },
}


XCEPTION_WEIGHTS_CONFIG = {
    "Xception41": {
        "tf_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception41_tf_in1k.weights.h5"
        },
    },
    "Xception41p": {
        "ra3_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception41p_ra3_in1k.weights.h5"
        },
    },
    "Xception65": {
        "ra3_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception65_ra3_in1k.weights.h5"
        },
        "tf_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception65_tf_in1k.weights.h5"
        },
    },
    "Xception65p": {
        "ra3_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception65p_ra3_in1k.weights.h5"
        },
    },
    "Xception71": {
        "tf_in1k": {
            "url": "https://github.com/IMvision12/keras-models/releases/download/classify3/xception71_tf_in1k.weights.h5"
        },
    },
}
