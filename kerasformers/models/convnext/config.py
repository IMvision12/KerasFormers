CONVNEXT_MODEL_CONFIG = {
    "convnext_atto": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [40, 80, 160, 320],
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "convnext_femto": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [48, 96, 192, 384],
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "convnext_pico": {
        "depths": [2, 2, 6, 2],
        "projection_dims": [64, 128, 256, 512],
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "convnext_nano": {
        "depths": [2, 2, 8, 2],
        "projection_dims": [80, 160, 320, 640],
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "convnext_tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "convnext_tiny_in22k": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
        "input_image_shape": 224,
        "num_classes": 21841,
    },
    "convnext_tiny_384": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
        "input_image_shape": 384,
        "num_classes": 1000,
    },
    "convnext_small": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "convnext_small_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
        "input_image_shape": 224,
        "num_classes": 21841,
    },
    "convnext_small_384": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
        "input_image_shape": 384,
        "num_classes": 1000,
    },
    "convnext_base": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "convnext_base_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
        "input_image_shape": 224,
        "num_classes": 21841,
    },
    "convnext_base_384": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
        "input_image_shape": 384,
        "num_classes": 1000,
    },
    "convnext_large": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "convnext_large_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
        "input_image_shape": 224,
        "num_classes": 21841,
    },
    "convnext_large_384": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
        "input_image_shape": 384,
        "num_classes": 1000,
    },
    "convnext_xlarge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
        "input_image_shape": 224,
        "num_classes": 1000,
    },
    "convnext_xlarge_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
        "input_image_shape": 224,
        "num_classes": 21841,
    },
    "convnext_xlarge_384": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
        "input_image_shape": 384,
        "num_classes": 1000,
    },
}

CONVNEXT_WEIGHT_CONFIG = {
    "convnext_atto_d2_in1k": {
        "model": "convnext_atto",
        "timm_id": "convnext_atto.d2_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_atto_d2_in1k.weights.h5",
    },
    "convnext_femto_d1_in1k": {
        "model": "convnext_femto",
        "timm_id": "convnext_femto.d1_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_femto_d1_in1k.weights.h5",
    },
    "convnext_pico_d1_in1k": {
        "model": "convnext_pico",
        "timm_id": "convnext_pico.d1_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_pico_d1_in1k.weights.h5",
    },
    "convnext_nano_d1h_in1k": {
        "model": "convnext_nano",
        "timm_id": "convnext_nano.d1h_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_nano_d1h_in1k.weights.h5",
    },
    "convnext_nano_in12k_ft_in1k": {
        "model": "convnext_nano",
        "timm_id": "convnext_nano.in12k_ft_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_nano_in12k_ft_in1k.weights.h5",
    },
    "convnext_tiny_fb_in1k": {
        "model": "convnext_tiny",
        "timm_id": "convnext_tiny.fb_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_tiny_fb_in1k.weights.h5",
    },
    "convnext_tiny_fb_in22k": {
        "model": "convnext_tiny_in22k",
        "timm_id": "convnext_tiny.fb_in22k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_tiny_fb_in22k.weights.h5",
    },
    "convnext_tiny_fb_in22k_ft_in1k": {
        "model": "convnext_tiny",
        "timm_id": "convnext_tiny.fb_in22k_ft_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_tiny_fb_in22k_ft_in1k.weights.h5",
    },
    "convnext_tiny_fb_in22k_ft_in1k_384": {
        "model": "convnext_tiny_384",
        "timm_id": "convnext_tiny.fb_in22k_ft_in1k_384",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_tiny_fb_in22k_ft_in1k_384.weights.h5",
    },
    "convnext_small_fb_in1k": {
        "model": "convnext_small",
        "timm_id": "convnext_small.fb_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_small_fb_in1k.weights.h5",
    },
    "convnext_small_fb_in22k": {
        "model": "convnext_small_in22k",
        "timm_id": "convnext_small.fb_in22k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_small_fb_in22k.weights.h5",
    },
    "convnext_small_fb_in22k_ft_in1k": {
        "model": "convnext_small",
        "timm_id": "convnext_small.fb_in22k_ft_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_small_fb_in22k_ft_in1k.weights.h5",
    },
    "convnext_small_fb_in22k_ft_in1k_384": {
        "model": "convnext_small_384",
        "timm_id": "convnext_small.fb_in22k_ft_in1k_384",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_small_fb_in22k_ft_in1k_384.weights.h5",
    },
    "convnext_base_fb_in1k": {
        "model": "convnext_base",
        "timm_id": "convnext_base.fb_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_base_fb_in1k.weights.h5",
    },
    "convnext_base_fb_in22k": {
        "model": "convnext_base_in22k",
        "timm_id": "convnext_base.fb_in22k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_base_fb_in22k.weights.h5",
    },
    "convnext_base_fb_in22k_ft_in1k": {
        "model": "convnext_base",
        "timm_id": "convnext_base.fb_in22k_ft_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_base_fb_in22k_ft_in1k.weights.h5",
    },
    "convnext_base_fb_in22k_ft_in1k_384": {
        "model": "convnext_base_384",
        "timm_id": "convnext_base.fb_in22k_ft_in1k_384",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_base_fb_in22k_ft_in1k_384.weights.h5",
    },
    "convnext_large_fb_in1k": {
        "model": "convnext_large",
        "timm_id": "convnext_large.fb_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_large_fb_in1k.weights.h5",
    },
    "convnext_large_fb_in22k": {
        "model": "convnext_large_in22k",
        "timm_id": "convnext_large.fb_in22k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_large_fb_in22k.weights.h5",
    },
    "convnext_large_fb_in22k_ft_in1k": {
        "model": "convnext_large",
        "timm_id": "convnext_large.fb_in22k_ft_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_large_fb_in22k_ft_in1k.weights.h5",
    },
    "convnext_large_fb_in22k_ft_in1k_384": {
        "model": "convnext_large_384",
        "timm_id": "convnext_large.fb_in22k_ft_in1k_384",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_large_fb_in22k_ft_in1k_384.weights.h5",
    },
    "convnext_xlarge_fb_in22k": {
        "model": "convnext_xlarge_in22k",
        "timm_id": "convnext_xlarge.fb_in22k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_xlarge_fb_in22k.weights.h5",
    },
    "convnext_xlarge_fb_in22k_ft_in1k": {
        "model": "convnext_xlarge",
        "timm_id": "convnext_xlarge.fb_in22k_ft_in1k",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_xlarge_fb_in22k_ft_in1k.weights.h5",
    },
    "convnext_xlarge_fb_in22k_ft_in1k_384": {
        "model": "convnext_xlarge_384",
        "timm_id": "convnext_xlarge.fb_in22k_ft_in1k_384",
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/classify1/convnext_xlarge_fb_in22k_ft_in1k_384.weights.h5",
    },
}
