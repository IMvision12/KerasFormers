from kerasformers.base import BaseConfig


class MobileViTV2Config(BaseConfig):
    r"""Configuration for [`MobileViTV2Model`] / [`MobileViTV2ImageClassify`].

    MobileViTV2 replaces MobileViT's multi-head attention with separable self-attention
    (linear in token count) and scales the whole network by a single width multiplier.
    One `kf_config.json` (declaring the canonical [`MobileViTV2ImageClassify`]) sits on
    each classification variant's repo, and both the backbone and classifier load from
    it. Fields mirror the model constructor and serialize flat.

    Args:
        multiplier (`float`, *optional*, defaults to 1.0):
            Width multiplier scaling all channel counts.
        image_size (`int`, *optional*, defaults to 256):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from kerasformers.models.mobilevitv2 import (
    ...     MobileViTV2Config,
    ...     MobileViTV2ImageClassify,
    ... )

    >>> configuration = MobileViTV2Config()
    >>> model = MobileViTV2ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mobilevitv2"

    multiplier: float = 1.0
    image_size: int = 256
    num_classes: int = 1000


# Hosted classification variants -> (model arch key, timm id). Weights load by Hub
# repo id (kf_config.json); the github release urls have been removed.
MOBILEVITV2_VARIANTS = {
    "mobilevitv2_050_cvnets_in1k": {
        "model": "mobilevitv2_050",
        "timm_id": "mobilevitv2_050.cvnets_in1k",
    },
    "mobilevitv2_075_cvnets_in1k": {
        "model": "mobilevitv2_075",
        "timm_id": "mobilevitv2_075.cvnets_in1k",
    },
    "mobilevitv2_100_cvnets_in1k": {
        "model": "mobilevitv2_100",
        "timm_id": "mobilevitv2_100.cvnets_in1k",
    },
    "mobilevitv2_125_cvnets_in1k": {
        "model": "mobilevitv2_125",
        "timm_id": "mobilevitv2_125.cvnets_in1k",
    },
    "mobilevitv2_150_cvnets_in1k": {
        "model": "mobilevitv2_150",
        "timm_id": "mobilevitv2_150.cvnets_in1k",
    },
    "mobilevitv2_150_cvnets_in22k_ft_in1k": {
        "model": "mobilevitv2_150",
        "timm_id": "mobilevitv2_150.cvnets_in22k_ft_in1k",
    },
    "mobilevitv2_150_cvnets_in22k_ft_in1k_384": {
        "model": "mobilevitv2_150_384",
        "timm_id": "mobilevitv2_150.cvnets_in22k_ft_in1k_384",
    },
    "mobilevitv2_175_cvnets_in1k": {
        "model": "mobilevitv2_175",
        "timm_id": "mobilevitv2_175.cvnets_in1k",
    },
    "mobilevitv2_175_cvnets_in22k_ft_in1k": {
        "model": "mobilevitv2_175",
        "timm_id": "mobilevitv2_175.cvnets_in22k_ft_in1k",
    },
    "mobilevitv2_175_cvnets_in22k_ft_in1k_384": {
        "model": "mobilevitv2_175_384",
        "timm_id": "mobilevitv2_175.cvnets_in22k_ft_in1k_384",
    },
    "mobilevitv2_200_cvnets_in1k": {
        "model": "mobilevitv2_200",
        "timm_id": "mobilevitv2_200.cvnets_in1k",
    },
    "mobilevitv2_200_cvnets_in22k_ft_in1k": {
        "model": "mobilevitv2_200",
        "timm_id": "mobilevitv2_200.cvnets_in22k_ft_in1k",
    },
    "mobilevitv2_200_cvnets_in22k_ft_in1k_384": {
        "model": "mobilevitv2_200_384",
        "timm_id": "mobilevitv2_200.cvnets_in22k_ft_in1k_384",
    },
}

MOBILEVITV2_SEGMENT_MODEL_CONFIG = {
    "mobilevitv2_100_deeplabv3": {
        "multiplier": 1.0,
        "image_size": 512,
        "num_classes": 21,
    },
    "mobilevitv2_150_deeplabv3": {
        "multiplier": 1.5,
        "image_size": 512,
        "num_classes": 21,
    },
}

MOBILEVITV2_SEGMENT_WEIGHTS_URLS = {
    "mobilevitv2_100_deeplabv3": {
        "model": "mobilevitv2_100_deeplabv3",
        "url": "https://huggingface.co/kerasformers/mobilevitv2_100_deeplabv3",
    },
    "mobilevitv2_150_deeplabv3": {
        "model": "mobilevitv2_150_deeplabv3",
        "url": "https://huggingface.co/kerasformers/mobilevitv2_150_deeplabv3",
    },
}
