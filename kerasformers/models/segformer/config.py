SEGFORMER_CONFIG = {
    "segformer_b0_cityscapes_1024": {
        "embed_dims": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "decode_head_dim": 256,
        "num_classes": 19,
        "input_image_shape": 1024,
    },
    "segformer_b0_cityscapes_768": {
        "embed_dims": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "decode_head_dim": 256,
        "num_classes": 19,
        "input_image_shape": 768,
    },
    "segformer_b0_ade_512": {
        "embed_dims": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "decode_head_dim": 256,
        "num_classes": 150,
        "input_image_shape": 512,
    },
    "segformer_b1_cityscapes_1024": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "decode_head_dim": 256,
        "num_classes": 19,
        "input_image_shape": 1024,
    },
    "segformer_b1_ade_512": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "decode_head_dim": 256,
        "num_classes": 150,
        "input_image_shape": 512,
    },
    "segformer_b2_cityscapes_1024": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "decode_head_dim": 768,
        "num_classes": 19,
        "input_image_shape": 1024,
    },
    "segformer_b2_ade_512": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "decode_head_dim": 768,
        "num_classes": 150,
        "input_image_shape": 512,
    },
    "segformer_b3_cityscapes_1024": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "decode_head_dim": 768,
        "num_classes": 19,
        "input_image_shape": 1024,
    },
    "segformer_b3_ade_512": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "decode_head_dim": 768,
        "num_classes": 150,
        "input_image_shape": 512,
    },
    "segformer_b4_cityscapes_1024": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 8, 27, 3],
        "decode_head_dim": 768,
        "num_classes": 19,
        "input_image_shape": 1024,
    },
    "segformer_b4_ade_512": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 8, 27, 3],
        "decode_head_dim": 768,
        "num_classes": 150,
        "input_image_shape": 512,
    },
    "segformer_b5_cityscapes_1024": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 6, 40, 3],
        "decode_head_dim": 768,
        "num_classes": 19,
        "input_image_shape": 1024,
    },
    "segformer_b5_ade_640": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 6, 40, 3],
        "decode_head_dim": 768,
        "num_classes": 150,
        "input_image_shape": 640,
    },
}

SEGFORMER_WEIGHTS = {
    "segformer_b0_cityscapes_1024": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b0_cityscapes_1024.weights.h5",
    },
    "segformer_b0_cityscapes_768": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b0_cityscapes_768.weights.h5",
    },
    "segformer_b0_ade_512": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b0_ade_512.weights.h5",
    },
    "segformer_b1_cityscapes_1024": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b1_cityscapes_1024.weights.h5",
    },
    "segformer_b1_ade_512": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b1_ade_512.weights.h5",
    },
    "segformer_b2_cityscapes_1024": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b2_cityscapes_1024.weights.h5",
    },
    "segformer_b2_ade_512": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b2_ade_512.weights.h5",
    },
    "segformer_b3_cityscapes_1024": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b3_cityscapes_1024.weights.h5",
    },
    "segformer_b3_ade_512": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b3_ade_512.weights.h5",
    },
    "segformer_b4_cityscapes_1024": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b4_cityscapes_1024.weights.h5",
    },
    "segformer_b4_ade_512": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b4_ade_512.weights.h5",
    },
    "segformer_b5_cityscapes_1024": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b5_cityscapes_1024.weights.h5",
    },
    "segformer_b5_ade_640": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/segformer/segformer_b5_ade_640.weights.h5",
    },
}
