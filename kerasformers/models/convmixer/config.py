CONVMIXER_MODEL_CONFIG = {
    "convmixer_1536_20": {
        "dim": 1536,
        "depth": 20,
        "patch_size": 7,
        "kernel_size": 9,
        "activation": "gelu",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convmixer_768_32": {
        "dim": 768,
        "depth": 32,
        "patch_size": 7,
        "kernel_size": 7,
        "activation": "relu",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convmixer_1024_20_ks9_p14": {
        "dim": 1024,
        "depth": 20,
        "patch_size": 14,
        "kernel_size": 9,
        "activation": "gelu",
        "image_size": 224,
        "num_classes": 1000,
    },
}

CONVMIXER_WEIGHT_CONFIG = {
    "convmixer_1536_20_in1k": {
        "model": "convmixer_1536_20",
        "timm_id": "convmixer_1536_20.in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/convmixer_1536_20_in1k.weights.h5",
    },
    "convmixer_768_32_in1k": {
        "model": "convmixer_768_32",
        "timm_id": "convmixer_768_32.in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/convmixer_768_32_in1k.weights.h5",
    },
    "convmixer_1024_20_ks9_p14_in1k": {
        "model": "convmixer_1024_20_ks9_p14",
        "timm_id": "convmixer_1024_20_ks9_p14.in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/v0.1/convmixer_1024_20_ks9_p14_in1k.weights.h5",
    },
}
