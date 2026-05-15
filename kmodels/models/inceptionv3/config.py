INCEPTIONV3_MODEL_CONFIG = {
    "inception_v3_tf_in1k": {
        "timm_id": "inception_v3.tf_in1k",
        "image_size": 299,
        "num_classes": 1000,
    },
    "inception_v3_tf_adv_in1k": {
        "timm_id": "inception_v3.tf_adv_in1k",
        "image_size": 299,
        "num_classes": 1000,
    },
    "inception_v3_gluon_in1k": {
        "timm_id": "inception_v3.gluon_in1k",
        "image_size": 299,
        "num_classes": 1000,
    },
}

INCEPTIONV3_WEIGHT_CONFIG = {
    "inception_v3_tf_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.1/inception_v3_tf_in1k.weights.h5",
    },
    "inception_v3_tf_adv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.1/inception_v3_tf_adv_in1k.weights.h5",
    },
    "inception_v3_gluon_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.1/inception_v3_gluon_in1k.weights.h5",
    },
}
