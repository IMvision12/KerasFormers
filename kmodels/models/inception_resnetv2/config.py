INCEPTION_RESNETV2_MODEL_CONFIG = {
    "inception_resnet_v2_tf_in1k": {
        "timm_id": "inception_resnet_v2.tf_in1k",
        "image_size": 299,
        "num_classes": 1000,
    },
    "inception_resnet_v2_tf_ens_adv_in1k": {
        "timm_id": "inception_resnet_v2.tf_ens_adv_in1k",
        "image_size": 299,
        "num_classes": 1000,
    },
}

INCEPTION_RESNETV2_WEIGHT_CONFIG = {
    "inception_resnet_v2_tf_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.1/inception_resnet_v2_tf_in1k.weights.h5",
    },
    "inception_resnet_v2_tf_ens_adv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.1/inception_resnet_v2_tf_ens_adv_in1k.weights.h5",
    },
}
