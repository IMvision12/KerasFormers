"""ResNet variant registry.

Variant ids follow timm naming: ``<arch>_<recipe>_<train_dataset>``.
``timm_id`` is the corresponding timm repo on HuggingFace Hub, used
by both the offline conversion script and the runtime ``timm:`` path
on :meth:`ResNet.from_timm`.
"""

_RESNET50 = {
    "block_repeats": [3, 4, 6, 3],
    "filters": [64, 128, 256, 512],
}
_RESNET101 = {
    "block_repeats": [3, 4, 23, 3],
    "filters": [64, 128, 256, 512],
}
_RESNET152 = {
    "block_repeats": [3, 8, 36, 3],
    "filters": [64, 128, 256, 512],
}


def _v(arch, timm_id):
    return {**arch, "timm_id": timm_id}


RESNET_CONFIG = {
    "resnet50_tv_in1k": _v(_RESNET50, "resnet50.tv_in1k"),
    "resnet50_a1_in1k": _v(_RESNET50, "resnet50.a1_in1k"),
    "resnet50_gluon_in1k": _v(_RESNET50, "resnet50.gluon_in1k"),
    "resnet101_tv_in1k": _v(_RESNET101, "resnet101.tv_in1k"),
    "resnet101_a1_in1k": _v(_RESNET101, "resnet101.a1_in1k"),
    "resnet101_gluon_in1k": _v(_RESNET101, "resnet101.gluon_in1k"),
    "resnet152_tv_in1k": _v(_RESNET152, "resnet152.tv_in1k"),
    "resnet152_a1_in1k": _v(_RESNET152, "resnet152.a1_in1k"),
    "resnet152_gluon_in1k": _v(_RESNET152, "resnet152.gluon_in1k"),
}

RESNET_WEIGHTS = {
    "resnet50_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet50_tv_in1k.weights.h5",
    },
    "resnet50_a1_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet50_a1_in1k.weights.h5",
    },
    "resnet50_gluon_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet50_gluon_in1k.weights.h5",
    },
    "resnet101_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet101_tv_in1k.weights.h5",
    },
    "resnet101_a1_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet101_a1_in1k.weights.h5",
    },
    "resnet101_gluon_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet101_gluon_in1k.weights.h5",
    },
    "resnet152_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet152_tv_in1k.weights.h5",
    },
    "resnet152_a1_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet152_a1_in1k.weights.h5",
    },
    "resnet152_gluon_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnet152_gluon_in1k.weights.h5",
    },
}
