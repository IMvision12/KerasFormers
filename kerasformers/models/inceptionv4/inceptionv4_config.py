INCEPTIONV4_MODEL_CONFIG = {
    "inception_v4": {
        "image_size": 299,
        "num_classes": 1000,
    },
}

INCEPTIONV4_WEIGHTS_URLS = {
    "inception_v4_tf_in1k": {
        "model": "inception_v4",
        "timm_id": "inception_v4.tf_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/inception_v4_tf_in1k.weights.h5",
    },
}
