MIT_MODEL_CONFIG = {
    "mit_b0": {
        "embed_dim": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "image_size": 224,
        "num_classes": 1000,
    },
    "mit_b1": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "image_size": 224,
        "num_classes": 1000,
    },
    "mit_b2": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "image_size": 224,
        "num_classes": 1000,
    },
    "mit_b3": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "image_size": 224,
        "num_classes": 1000,
    },
    "mit_b4": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 8, 27, 3],
        "image_size": 224,
        "num_classes": 1000,
    },
    "mit_b5": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 6, 40, 3],
        "image_size": 224,
        "num_classes": 1000,
    },
}

MIT_WEIGHTS_URLS = {
    "mit_b0_in1k": {
        "model": "mit_b0",
        "hf_id": "nvidia/mit-b0",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/mit_b0_in1k.weights.h5",
    },
    "mit_b1_in1k": {
        "model": "mit_b1",
        "hf_id": "nvidia/mit-b1",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/mit_b1_in1k.weights.h5",
    },
    "mit_b2_in1k": {
        "model": "mit_b2",
        "hf_id": "nvidia/mit-b2",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/mit_b2_in1k.weights.h5",
    },
    "mit_b3_in1k": {
        "model": "mit_b3",
        "hf_id": "nvidia/mit-b3",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/mit_b3_in1k.weights.h5",
    },
    "mit_b4_in1k": {
        "model": "mit_b4",
        "hf_id": "nvidia/mit-b4",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/mit_b4_in1k.weights.h5",
    },
    "mit_b5_in1k": {
        "model": "mit_b5",
        "hf_id": "nvidia/mit-b5",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/mit_b5_in1k.weights.h5",
    },
}
