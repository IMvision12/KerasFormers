"""InceptionResNetV2 variant registry (timm-ported)."""

_INCEPTION_RESNET_V2 = {}  # Fixed architecture.


def _v(arch, timm_id, image_size=299, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


INCEPTION_RESNET_V2_CONFIG = {
    "inception_resnet_v2_tf_in1k": _v(
        _INCEPTION_RESNET_V2, "inception_resnet_v2.tf_in1k"
    ),
    "inception_resnet_v2_tf_ens_adv_in1k": _v(
        _INCEPTION_RESNET_V2, "inception_resnet_v2.tf_ens_adv_in1k"
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
INCEPTION_RESNET_V2_WEIGHTS = {
    "inception_resnet_v2_tf_in1k": {
        "url": f"{_BASE_URL}/inception_resnet_v2_tf_in1k.weights.h5"
    },
    "inception_resnet_v2_tf_ens_adv_in1k": {
        "url": f"{_BASE_URL}/inception_resnet_v2_tf_ens_adv_in1k.weights.h5"
    },
}
