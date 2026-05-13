"""VGG variant registry (timm-ported)."""

_VGG11 = {
    "num_filters": [
        64,
        "M",
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "batch_norm": False,
}
_VGG11_BN = {**_VGG11, "batch_norm": True}

_VGG13 = {
    "num_filters": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "batch_norm": False,
}
_VGG13_BN = {**_VGG13, "batch_norm": True}

_VGG16 = {
    "num_filters": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "batch_norm": False,
}
_VGG16_BN = {**_VGG16, "batch_norm": True}

_VGG19 = {
    "num_filters": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
    "batch_norm": False,
}
_VGG19_BN = {**_VGG19, "batch_norm": True}


def _v(arch, timm_id, image_size=224, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


VGG_CONFIG = {
    "vgg11_tv_in1k": _v(_VGG11, "vgg11.tv_in1k"),
    "vgg11_bn_tv_in1k": _v(_VGG11_BN, "vgg11_bn.tv_in1k"),
    "vgg13_tv_in1k": _v(_VGG13, "vgg13.tv_in1k"),
    "vgg13_bn_tv_in1k": _v(_VGG13_BN, "vgg13_bn.tv_in1k"),
    "vgg16_tv_in1k": _v(_VGG16, "vgg16.tv_in1k"),
    "vgg16_bn_tv_in1k": _v(_VGG16_BN, "vgg16_bn.tv_in1k"),
    "vgg19_tv_in1k": _v(_VGG19, "vgg19.tv_in1k"),
    "vgg19_bn_tv_in1k": _v(_VGG19_BN, "vgg19_bn.tv_in1k"),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
VGG_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in VGG_CONFIG
}
