CONVNEXT_MODEL_CONFIG = {
    "convnext_atto_d2_in1k": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [40, 80, 160, 320],
        "timm_id": "convnext_atto.d2_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_femto_d1_in1k": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [48, 96, 192, 384],
        "timm_id": "convnext_femto.d1_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_pico_d1_in1k": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [64, 128, 256, 512],
        "timm_id": "convnext_pico.d1_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_nano_d1h_in1k": {
        "depths": [2, 2, 8, 2],
        "projection_dims": [80, 160, 320, 640],
        "timm_id": "convnext_nano.d1h_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_nano_in12k_ft_in1k": {
        "depths": [2, 2, 8, 2],
        "projection_dims": [80, 160, 320, 640],
        "timm_id": "convnext_nano.in12k_ft_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_tiny_fb_in1k": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
        "timm_id": "convnext_tiny.fb_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_tiny_fb_in22k": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
        "timm_id": "convnext_tiny.fb_in22k",
        "image_size": 224,
        "num_classes": 21841,
    },
    "convnext_tiny_fb_in22k_ft_in1k": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
        "timm_id": "convnext_tiny.fb_in22k_ft_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_tiny_fb_in22k_ft_in1k_384": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
        "timm_id": "convnext_tiny.fb_in22k_ft_in1k_384",
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnext_small_fb_in1k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
        "timm_id": "convnext_small.fb_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_small_fb_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
        "timm_id": "convnext_small.fb_in22k",
        "image_size": 224,
        "num_classes": 21841,
    },
    "convnext_small_fb_in22k_ft_in1k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
        "timm_id": "convnext_small.fb_in22k_ft_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_small_fb_in22k_ft_in1k_384": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
        "timm_id": "convnext_small.fb_in22k_ft_in1k_384",
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnext_base_fb_in1k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
        "timm_id": "convnext_base.fb_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_base_fb_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
        "timm_id": "convnext_base.fb_in22k",
        "image_size": 224,
        "num_classes": 21841,
    },
    "convnext_base_fb_in22k_ft_in1k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
        "timm_id": "convnext_base.fb_in22k_ft_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_base_fb_in22k_ft_in1k_384": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
        "timm_id": "convnext_base.fb_in22k_ft_in1k_384",
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnext_large_fb_in1k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
        "timm_id": "convnext_large.fb_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_large_fb_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
        "timm_id": "convnext_large.fb_in22k",
        "image_size": 224,
        "num_classes": 21841,
    },
    "convnext_large_fb_in22k_ft_in1k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
        "timm_id": "convnext_large.fb_in22k_ft_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_large_fb_in22k_ft_in1k_384": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
        "timm_id": "convnext_large.fb_in22k_ft_in1k_384",
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnext_xlarge_fb_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
        "timm_id": "convnext_xlarge.fb_in22k",
        "image_size": 224,
        "num_classes": 21841,
    },
    "convnext_xlarge_fb_in22k_ft_in1k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
        "timm_id": "convnext_xlarge.fb_in22k_ft_in1k",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_xlarge_fb_in22k_ft_in1k_384": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
        "timm_id": "convnext_xlarge.fb_in22k_ft_in1k_384",
        "image_size": 384,
        "num_classes": 1000,
    },
}

CONVNEXT_WEIGHT_CONFIG = {
    "convnext_atto_d2_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_atto_d2_in1k.weights.h5",
    },
    "convnext_femto_d1_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_femto_d1_in1k.weights.h5",
    },
    "convnext_pico_d1_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_pico_d1_in1k.weights.h5",
    },
    "convnext_nano_d1h_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_nano_d1h_in1k.weights.h5",
    },
    "convnext_nano_in12k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_nano_in12k_ft_in1k.weights.h5",
    },
    "convnext_tiny_fb_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_tiny_fb_in1k.weights.h5",
    },
    "convnext_tiny_fb_in22k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_tiny_fb_in22k.weights.h5",
    },
    "convnext_tiny_fb_in22k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_tiny_fb_in22k_ft_in1k.weights.h5",
    },
    "convnext_tiny_fb_in22k_ft_in1k_384": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_tiny_fb_in22k_ft_in1k_384.weights.h5",
    },
    "convnext_small_fb_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_small_fb_in1k.weights.h5",
    },
    "convnext_small_fb_in22k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_small_fb_in22k.weights.h5",
    },
    "convnext_small_fb_in22k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_small_fb_in22k_ft_in1k.weights.h5",
    },
    "convnext_small_fb_in22k_ft_in1k_384": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_small_fb_in22k_ft_in1k_384.weights.h5",
    },
    "convnext_base_fb_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_base_fb_in1k.weights.h5",
    },
    "convnext_base_fb_in22k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_base_fb_in22k.weights.h5",
    },
    "convnext_base_fb_in22k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_base_fb_in22k_ft_in1k.weights.h5",
    },
    "convnext_base_fb_in22k_ft_in1k_384": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_base_fb_in22k_ft_in1k_384.weights.h5",
    },
    "convnext_large_fb_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_large_fb_in1k.weights.h5",
    },
    "convnext_large_fb_in22k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_large_fb_in22k.weights.h5",
    },
    "convnext_large_fb_in22k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_large_fb_in22k_ft_in1k.weights.h5",
    },
    "convnext_large_fb_in22k_ft_in1k_384": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_large_fb_in22k_ft_in1k_384.weights.h5",
    },
    "convnext_xlarge_fb_in22k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_xlarge_fb_in22k.weights.h5",
    },
    "convnext_xlarge_fb_in22k_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_xlarge_fb_in22k_ft_in1k.weights.h5",
    },
    "convnext_xlarge_fb_in22k_ft_in1k_384": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/convnext/convnext_xlarge_fb_in22k_ft_in1k_384.weights.h5",
    },
}
