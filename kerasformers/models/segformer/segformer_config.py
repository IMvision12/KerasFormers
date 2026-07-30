SEGFORMER_CONFIG = {
    "segformer_b0_cityscapes_1024": {
        "embed_dim": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "decode_head_dim": 256,
        "num_classes": 19,
        "image_size": 1024,
    },
    "segformer_b0_cityscapes_768": {
        "embed_dim": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "decode_head_dim": 256,
        "num_classes": 19,
        "image_size": 768,
    },
    "segformer_b0_ade_512": {
        "embed_dim": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "decode_head_dim": 256,
        "num_classes": 150,
        "image_size": 512,
    },
    "segformer_b1_cityscapes_1024": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "decode_head_dim": 256,
        "num_classes": 19,
        "image_size": 1024,
    },
    "segformer_b1_ade_512": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "decode_head_dim": 256,
        "num_classes": 150,
        "image_size": 512,
    },
    "segformer_b2_cityscapes_1024": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "decode_head_dim": 768,
        "num_classes": 19,
        "image_size": 1024,
    },
    "segformer_b2_ade_512": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "decode_head_dim": 768,
        "num_classes": 150,
        "image_size": 512,
    },
    "segformer_b3_cityscapes_1024": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "decode_head_dim": 768,
        "num_classes": 19,
        "image_size": 1024,
    },
    "segformer_b3_ade_512": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "decode_head_dim": 768,
        "num_classes": 150,
        "image_size": 512,
    },
    "segformer_b4_cityscapes_1024": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 8, 27, 3],
        "decode_head_dim": 768,
        "num_classes": 19,
        "image_size": 1024,
    },
    "segformer_b4_ade_512": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 8, 27, 3],
        "decode_head_dim": 768,
        "num_classes": 150,
        "image_size": 512,
    },
    "segformer_b5_cityscapes_1024": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 6, 40, 3],
        "decode_head_dim": 768,
        "num_classes": 19,
        "image_size": 1024,
    },
    "segformer_b5_ade_640": {
        "embed_dim": [64, 128, 320, 512],
        "depths": [3, 6, 40, 3],
        "decode_head_dim": 768,
        "num_classes": 150,
        "image_size": 640,
    },
}

SEGFORMER_WEIGHTS_URLS = {
    "segformer_b0_cityscapes_1024": {
        "url": "https://huggingface.co/kerasformers/segformer_b0_cityscapes_1024",
    },
    "segformer_b0_cityscapes_768": {
        "url": "https://huggingface.co/kerasformers/segformer_b0_cityscapes_768",
    },
    "segformer_b0_ade_512": {
        "url": "https://huggingface.co/kerasformers/segformer_b0_ade_512",
    },
    "segformer_b1_cityscapes_1024": {
        "url": "https://huggingface.co/kerasformers/segformer_b1_cityscapes_1024",
    },
    "segformer_b1_ade_512": {
        "url": "https://huggingface.co/kerasformers/segformer_b1_ade_512",
    },
    "segformer_b2_cityscapes_1024": {
        "url": "https://huggingface.co/kerasformers/segformer_b2_cityscapes_1024",
    },
    "segformer_b2_ade_512": {
        "url": "https://huggingface.co/kerasformers/segformer_b2_ade_512",
    },
    "segformer_b3_cityscapes_1024": {
        "url": "https://huggingface.co/kerasformers/segformer_b3_cityscapes_1024",
    },
    "segformer_b3_ade_512": {
        "url": "https://huggingface.co/kerasformers/segformer_b3_ade_512",
    },
    "segformer_b4_cityscapes_1024": {
        "url": "https://huggingface.co/kerasformers/segformer_b4_cityscapes_1024",
    },
    "segformer_b4_ade_512": {
        "url": "https://huggingface.co/kerasformers/segformer_b4_ade_512",
    },
    "segformer_b5_cityscapes_1024": {
        "url": "https://huggingface.co/kerasformers/segformer_b5_cityscapes_1024",
    },
    "segformer_b5_ade_640": {
        "url": "https://huggingface.co/kerasformers/segformer_b5_ade_640",
    },
}
