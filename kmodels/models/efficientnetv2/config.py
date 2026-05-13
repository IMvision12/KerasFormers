"""EfficientNetV2 variant registry (timm-ported)."""

EFFICIENTNETV2_BLOCK_CONFIG = {
    "EfficientNetV2S": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 48,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-6: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 6,
            "input_filters": 64,
            "output_filters": 128,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 9,
            "input_filters": 128,
            "output_filters": 160,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 15,
            "input_filters": 160,
            "output_filters": 256,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "EfficientNetV2M": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 48,
            "output_filters": 80,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-7: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 80,
            "output_filters": 160,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 14,
            "input_filters": 160,
            "output_filters": 176,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 18,
            "input_filters": 176,
            "output_filters": 304,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 304,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "EfficientNetV2L": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-7: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 10,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 19,
            "input_filters": 192,
            "output_filters": 224,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 25,
            "input_filters": 224,
            "output_filters": 384,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 384,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "EfficientNetV2XL": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-7: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 16,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 24,
            "input_filters": 192,
            "output_filters": 256,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 32,
            "input_filters": 256,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 512,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    # Shared block config for all B variants (B0, B1, B2, B3)
    "EfficientNetV2B": [
        # Stage 1: Initial stage
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        # Stage 2-3: Early stages with no SE
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        # Stage 4-6: Later stages with SE
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1.0 / 3.0,
        "mode": "fan_out",
        "distribution": "uniform",
    },
}


# Architecture kwargs per family (size/coefficients/block_arch_key/head_filters).
_S = {
    "width_coefficient": 1.0,
    "depth_coefficient": 1.0,
    "default_size": 300,
    "block_arch": "EfficientNetV2S",
    "head_filters": 1280,
}
_M = {
    "width_coefficient": 1.0,
    "depth_coefficient": 1.0,
    "default_size": 384,
    "block_arch": "EfficientNetV2M",
    "head_filters": 1280,
}
_L = {
    "width_coefficient": 1.0,
    "depth_coefficient": 1.0,
    "default_size": 384,
    "block_arch": "EfficientNetV2L",
    "head_filters": 1280,
}
_XL = {
    "width_coefficient": 1.0,
    "depth_coefficient": 1.0,
    "default_size": 384,
    "block_arch": "EfficientNetV2XL",
    "head_filters": 1280,
}
_B0 = {
    "width_coefficient": 1.0,
    "depth_coefficient": 1.0,
    "default_size": 192,
    "block_arch": "EfficientNetV2B",
    "head_filters": 1280,
}
_B1 = {
    "width_coefficient": 1.0,
    "depth_coefficient": 1.1,
    "default_size": 192,
    "block_arch": "EfficientNetV2B",
    "head_filters": 1280,
}
_B2 = {
    "width_coefficient": 1.1,
    "depth_coefficient": 1.2,
    "default_size": 208,
    "block_arch": "EfficientNetV2B",
    "head_filters": 1408,
}
_B3 = {
    "width_coefficient": 1.2,
    "depth_coefficient": 1.4,
    "default_size": 240,
    "block_arch": "EfficientNetV2B",
    "head_filters": 1536,
}


def _v(arch, timm_id, image_size, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


EFFICIENTNETV2_CONFIG = {
    # S
    "tf_efficientnetv2_s_in1k": _v(_S, "tf_efficientnetv2_s.in1k", 300),
    "tf_efficientnetv2_s_in21k": _v(
        _S, "tf_efficientnetv2_s.in21k", 300, num_classes=21843
    ),
    "tf_efficientnetv2_s_in21k_ft_in1k": _v(
        _S, "tf_efficientnetv2_s.in21k_ft_in1k", 300
    ),
    # M
    "tf_efficientnetv2_m_in1k": _v(_M, "tf_efficientnetv2_m.in1k", 384),
    "tf_efficientnetv2_m_in21k": _v(
        _M, "tf_efficientnetv2_m.in21k", 384, num_classes=21843
    ),
    "tf_efficientnetv2_m_in21k_ft_in1k": _v(
        _M, "tf_efficientnetv2_m.in21k_ft_in1k", 384
    ),
    # L
    "tf_efficientnetv2_l_in1k": _v(_L, "tf_efficientnetv2_l.in1k", 384),
    "tf_efficientnetv2_l_in21k": _v(
        _L, "tf_efficientnetv2_l.in21k", 384, num_classes=21843
    ),
    "tf_efficientnetv2_l_in21k_ft_in1k": _v(
        _L, "tf_efficientnetv2_l.in21k_ft_in1k", 384
    ),
    # XL
    "tf_efficientnetv2_xl_in21k": _v(
        _XL, "tf_efficientnetv2_xl.in21k", 384, num_classes=21843
    ),
    "tf_efficientnetv2_xl_in21k_ft_in1k": _v(
        _XL, "tf_efficientnetv2_xl.in21k_ft_in1k", 384
    ),
    # B0
    "tf_efficientnetv2_b0_in1k": _v(_B0, "tf_efficientnetv2_b0.in1k", 192),
    # B1
    "tf_efficientnetv2_b1_in1k": _v(_B1, "tf_efficientnetv2_b1.in1k", 192),
    # B2
    "tf_efficientnetv2_b2_in1k": _v(_B2, "tf_efficientnetv2_b2.in1k", 208),
    # B3
    "tf_efficientnetv2_b3_in1k": _v(_B3, "tf_efficientnetv2_b3.in1k", 240),
    "tf_efficientnetv2_b3_in21k_ft_in1k": _v(
        _B3, "tf_efficientnetv2_b3.in21k_ft_in1k", 240
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
EFFICIENTNETV2_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"}
    for variant in EFFICIENTNETV2_CONFIG
}
