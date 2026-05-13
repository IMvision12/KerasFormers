"""InceptionV3 variant registry (timm-ported)."""

_INCEPTIONV3 = {}  # No arch kwargs; InceptionV3 is a fixed architecture.


def _v(arch, timm_id, image_size=299, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


INCEPTIONV3_CONFIG = {
    "inception_v3_tf_in1k": _v(_INCEPTIONV3, "inception_v3.tf_in1k"),
    "inception_v3_tf_adv_in1k": _v(_INCEPTIONV3, "inception_v3.tf_adv_in1k"),
    "inception_v3_gluon_in1k": _v(_INCEPTIONV3, "inception_v3.gluon_in1k"),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
INCEPTIONV3_WEIGHTS = {
    "inception_v3_tf_in1k": {"url": f"{_BASE_URL}/inception_v3_tf_in1k.weights.h5"},
    "inception_v3_tf_adv_in1k": {
        "url": f"{_BASE_URL}/inception_v3_tf_adv_in1k.weights.h5"
    },
    "inception_v3_gluon_in1k": {
        "url": f"{_BASE_URL}/inception_v3_gluon_in1k.weights.h5"
    },
}
