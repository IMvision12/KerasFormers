EFFICIENTNET_LITE_MODEL_CONFIG = {
    "efficientnet_lite_b0": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "dropout_rate": 0.2,
        "default_size": 224,
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "efficientnet_lite_b1": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.1,
        "dropout_rate": 0.2,
        "default_size": 240,
        "input_image_shape": 240,
        "num_classes": 1000,
    },
    "efficientnet_lite_b2": {
        "width_coefficient": 1.1,
        "depth_coefficient": 1.2,
        "dropout_rate": 0.3,
        "default_size": 260,
        "input_image_shape": 260,
        "num_classes": 1000,
    },
    "efficientnet_lite_b3": {
        "width_coefficient": 1.2,
        "depth_coefficient": 1.4,
        "dropout_rate": 0.3,
        "default_size": 300,
        "input_image_shape": 300,
        "num_classes": 1000,
    },
    "efficientnet_lite_b4": {
        "width_coefficient": 1.4,
        "depth_coefficient": 1.8,
        "dropout_rate": 0.3,
        "default_size": 380,
        "input_image_shape": 380,
        "num_classes": 1000,
    },
}

EFFICIENTNET_LITE_WEIGHT_CONFIG = {
    "tf_efficientnet_lite0_in1k": {
        "model": "efficientnet_lite_b0",
        "timm_id": "tf_efficientnet_lite0.in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/tf_efficientnet_lite0_in1k.weights.h5",
    },
    "tf_efficientnet_lite1_in1k": {
        "model": "efficientnet_lite_b1",
        "timm_id": "tf_efficientnet_lite1.in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/tf_efficientnet_lite1_in1k.weights.h5",
    },
    "tf_efficientnet_lite2_in1k": {
        "model": "efficientnet_lite_b2",
        "timm_id": "tf_efficientnet_lite2.in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/tf_efficientnet_lite2_in1k.weights.h5",
    },
    "tf_efficientnet_lite3_in1k": {
        "model": "efficientnet_lite_b3",
        "timm_id": "tf_efficientnet_lite3.in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/tf_efficientnet_lite3_in1k.weights.h5",
    },
    "tf_efficientnet_lite4_in1k": {
        "model": "efficientnet_lite_b4",
        "timm_id": "tf_efficientnet_lite4.in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify2/tf_efficientnet_lite4_in1k.weights.h5",
    },
}
