_MIT_B0 = {"embed_dims": [32, 64, 160, 256], "depths": [2, 2, 2, 2]}
_MIT_B1 = {"embed_dims": [64, 128, 320, 512], "depths": [2, 2, 2, 2]}
_MIT_B2 = {"embed_dims": [64, 128, 320, 512], "depths": [3, 4, 6, 3]}
_MIT_B3 = {"embed_dims": [64, 128, 320, 512], "depths": [3, 4, 18, 3]}
_MIT_B4 = {"embed_dims": [64, 128, 320, 512], "depths": [3, 8, 27, 3]}
_MIT_B5 = {"embed_dims": [64, 128, 320, 512], "depths": [3, 6, 40, 3]}

_DECODE_HEAD_DIM_SMALL = 256
_DECODE_HEAD_DIM_LARGE = 768


SEGFORMER_CONFIG = {
    "segformer_b0_cityscapes_1024": {
        **_MIT_B0,
        "decode_head_dim": _DECODE_HEAD_DIM_SMALL,
        "num_classes": 19,
        "input_shape": (1024, 1024, 3),
    },
    "segformer_b0_cityscapes_768": {
        **_MIT_B0,
        "decode_head_dim": _DECODE_HEAD_DIM_SMALL,
        "num_classes": 19,
        "input_shape": (768, 768, 3),
    },
    "segformer_b0_ade_512": {
        **_MIT_B0,
        "decode_head_dim": _DECODE_HEAD_DIM_SMALL,
        "num_classes": 150,
        "input_shape": (512, 512, 3),
    },
    "segformer_b1_cityscapes_1024": {
        **_MIT_B1,
        "decode_head_dim": _DECODE_HEAD_DIM_SMALL,
        "num_classes": 19,
        "input_shape": (1024, 1024, 3),
    },
    "segformer_b1_ade_512": {
        **_MIT_B1,
        "decode_head_dim": _DECODE_HEAD_DIM_SMALL,
        "num_classes": 150,
        "input_shape": (512, 512, 3),
    },
    "segformer_b2_cityscapes_1024": {
        **_MIT_B2,
        "decode_head_dim": _DECODE_HEAD_DIM_LARGE,
        "num_classes": 19,
        "input_shape": (1024, 1024, 3),
    },
    "segformer_b2_ade_512": {
        **_MIT_B2,
        "decode_head_dim": _DECODE_HEAD_DIM_LARGE,
        "num_classes": 150,
        "input_shape": (512, 512, 3),
    },
    "segformer_b3_cityscapes_1024": {
        **_MIT_B3,
        "decode_head_dim": _DECODE_HEAD_DIM_LARGE,
        "num_classes": 19,
        "input_shape": (1024, 1024, 3),
    },
    "segformer_b3_ade_512": {
        **_MIT_B3,
        "decode_head_dim": _DECODE_HEAD_DIM_LARGE,
        "num_classes": 150,
        "input_shape": (512, 512, 3),
    },
    "segformer_b4_cityscapes_1024": {
        **_MIT_B4,
        "decode_head_dim": _DECODE_HEAD_DIM_LARGE,
        "num_classes": 19,
        "input_shape": (1024, 1024, 3),
    },
    "segformer_b4_ade_512": {
        **_MIT_B4,
        "decode_head_dim": _DECODE_HEAD_DIM_LARGE,
        "num_classes": 150,
        "input_shape": (512, 512, 3),
    },
    "segformer_b5_cityscapes_1024": {
        **_MIT_B5,
        "decode_head_dim": _DECODE_HEAD_DIM_LARGE,
        "num_classes": 19,
        "input_shape": (1024, 1024, 3),
    },
    "segformer_b5_ade_640": {
        **_MIT_B5,
        "decode_head_dim": _DECODE_HEAD_DIM_LARGE,
        "num_classes": 150,
        "input_shape": (640, 640, 3),
    },
}

SEGFORMER_WEIGHTS = {
    "segformer_b0_cityscapes_1024": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b0_cityscapes_1024.weights.h5",
    },
    "segformer_b0_cityscapes_768": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b0_cityscapes_768.weights.h5",
    },
    "segformer_b0_ade_512": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b0_ade_512.weights.h5",
    },
    "segformer_b1_cityscapes_1024": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b1_cityscapes_1024.weights.h5",
    },
    "segformer_b1_ade_512": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b1_ade_512.weights.h5",
    },
    "segformer_b2_cityscapes_1024": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b2_cityscapes_1024.weights.h5",
    },
    "segformer_b2_ade_512": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b2_ade_512.weights.h5",
    },
    "segformer_b3_cityscapes_1024": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b3_cityscapes_1024.weights.h5",
    },
    "segformer_b3_ade_512": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b3_ade_512.weights.h5",
    },
    "segformer_b4_cityscapes_1024": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b4_cityscapes_1024.weights.h5",
    },
    "segformer_b4_ade_512": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b4_ade_512.weights.h5",
    },
    "segformer_b5_cityscapes_1024": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b5_cityscapes_1024.weights.h5",
    },
    "segformer_b5_ade_640": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/segformer/segformer_b5_ade_640.weights.h5",
    },
}
