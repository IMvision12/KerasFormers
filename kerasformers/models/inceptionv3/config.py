INCEPTIONV3_MODEL_CONFIG = {
    "inception_v3": {
        "input_image_shape": 299,
        "num_classes": 1000,
    },
}

INCEPTIONV3_WEIGHT_CONFIG = {
    "inception_v3_tf_in1k": {
        "model": "inception_v3",
        "timm_id": "inception_v3.tf_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/inception_v3_tf_in1k.weights.h5",
    },
    "inception_v3_tf_adv_in1k": {
        "model": "inception_v3",
        "timm_id": "inception_v3.tf_adv_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/inception_v3_tf_adv_in1k.weights.h5",
    },
    "inception_v3_gluon_in1k": {
        "model": "inception_v3",
        "timm_id": "inception_v3.gluon_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/inception_v3_gluon_in1k.weights.h5",
    },
}
