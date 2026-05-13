"""InceptionV4 variant registry (timm-ported)."""

_INCEPTIONV4 = {}  # No arch kwargs; InceptionV4 is a fixed architecture.


def _v(arch, timm_id, image_size=299, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


INCEPTIONV4_CONFIG = {
    "inception_v4_tf_in1k": _v(_INCEPTIONV4, "inception_v4.tf_in1k"),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
INCEPTIONV4_WEIGHTS = {
    "inception_v4_tf_in1k": {"url": f"{_BASE_URL}/inception_v4.weights.h5"},
}
