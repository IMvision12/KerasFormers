RESNEXT_MODEL_CONFIG = {
    "resnext50_32x4d_a1_in1k": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "timm_id": "resnext50_32x4d.a1_in1k",
    },
    "resnext50_32x4d_tv_in1k": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "timm_id": "resnext50_32x4d.tv_in1k",
    },
    "resnext50_32x4d_gluon_in1k": {
        "block_repeats": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "timm_id": "resnext50_32x4d.gluon_in1k",
    },
    "resnext101_32x4d_gluon_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "timm_id": "resnext101_32x4d.gluon_in1k",
    },
    "resnext101_32x4d_fb_ssl_yfcc100m_ft_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "timm_id": "resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k",
    },
    "resnext101_32x4d_fb_swsl_ig1b_ft_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "timm_id": "resnext101_32x4d.fb_swsl_ig1b_ft_in1k",
    },
    "resnext101_32x8d_tv_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 4,
        "timm_id": "resnext101_32x8d.tv_in1k",
    },
    "resnext101_32x8d_fb_wsl_ig1b_ft_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 4,
        "timm_id": "resnext101_32x8d.fb_wsl_ig1b_ft_in1k",
    },
    "resnext101_32x8d_fb_ssl_yfcc100m_ft_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 4,
        "timm_id": "resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k",
    },
    "resnext101_32x8d_fb_swsl_ig1b_ft_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 4,
        "timm_id": "resnext101_32x8d.fb_swsl_ig1b_ft_in1k",
    },
    "resnext101_32x16d_fb_wsl_ig1b_ft_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 8,
        "timm_id": "resnext101_32x16d.fb_wsl_ig1b_ft_in1k",
    },
    "resnext101_32x16d_fb_ssl_yfcc100m_ft_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 8,
        "timm_id": "resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k",
    },
    "resnext101_32x16d_fb_swsl_ig1b_ft_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 8,
        "timm_id": "resnext101_32x16d.fb_swsl_ig1b_ft_in1k",
    },
    "resnext101_32x32d_fb_wsl_ig1b_ft_in1k": {
        "block_repeats": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 16,
        "timm_id": "resnext101_32x32d.fb_wsl_ig1b_ft_in1k",
    },
}

RESNEXT_WEIGHT_CONFIG = {
    "resnext50_32x4d_a1_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext50_32x4d_a1_in1k.weights.h5",
    },
    "resnext50_32x4d_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext50_32x4d_tv_in1k.weights.h5",
    },
    "resnext50_32x4d_gluon_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext50_32x4d_gluon_in1k.weights.h5",
    },
    "resnext101_32x4d_gluon_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext101_32x4d_gluon_in1k.weights.h5",
    },
    "resnext101_32x4d_fb_ssl_yfcc100m_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext101_32x4d_fb_ssl_yfcc100m_ft_in1k.weights.h5",
    },
    "resnext101_32x4d_fb_swsl_ig1b_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext101_32x4d_fb_swsl_ig1b_ft_in1k.weights.h5",
    },
    "resnext101_32x8d_tv_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext101_32x8d_tv_in1k.weights.h5",
    },
    "resnext101_32x8d_fb_wsl_ig1b_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext101_32x8d_fb_wsl_ig1b_ft_in1k.weights.h5",
    },
    "resnext101_32x8d_fb_ssl_yfcc100m_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext101_32x8d_fb_ssl_yfcc100m_ft_in1k.weights.h5",
    },
    "resnext101_32x8d_fb_swsl_ig1b_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext101_32x8d_fb_swsl_ig1b_ft_in1k.weights.h5",
    },
    "resnext101_32x16d_fb_wsl_ig1b_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext101_32x16d_fb_wsl_ig1b_ft_in1k.weights.h5",
    },
    "resnext101_32x16d_fb_ssl_yfcc100m_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext101_32x16d_fb_ssl_yfcc100m_ft_in1k.weights.h5",
    },
    "resnext101_32x16d_fb_swsl_ig1b_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext101_32x16d_fb_swsl_ig1b_ft_in1k.weights.h5",
    },
    "resnext101_32x32d_fb_wsl_ig1b_ft_in1k": {
        "url": "https://github.com/IMvision12/keras-models/releases/download/v0.2/resnext101_32x32d_fb_wsl_ig1b_ft_in1k.weights.h5",
    },
}
