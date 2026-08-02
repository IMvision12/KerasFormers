from kerasformers.base import BaseConfig


class MobileViTConfig(BaseConfig):
    r"""Configuration for [`MobileViTModel`] / [`MobileViTImageClassify`].

    MobileViT interleaves MobileNetV2 blocks with lightweight transformer blocks that
    apply global attention over unfolded patches, giving a compact conv/transformer
    hybrid. One `kf_config.json` (declaring the canonical [`MobileViTImageClassify`])
    sits on each classification variant's repo, and both the backbone and classifier
    load from it. Fields mirror the model constructor and serialize flat.

    Args:
        initial_dims (`int`, *optional*, defaults to 16):
            Channel count of the stem convolution.
        head_dims (`int`, *optional*, defaults to 640):
            Channel count of the final 1x1 head convolution.
        block_dims (`tuple`, *optional*, defaults to `(32, 64, 96, 128, 160)`):
            Output channel count per stage.
        expansion_ratio (`tuple`, *optional*, defaults to `(4.0, 4.0, 4.0, 4.0, 4.0)`):
            Inverted-residual expansion ratio per stage.
        attention_dims (`tuple`, *optional*, defaults to `(None, None, 144, 192, 240)`):
            Transformer hidden size per stage; `None` marks the purely convolutional
            early stages.
        image_size (`int`, *optional*, defaults to 256):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from kerasformers.models.mobilevit import (
    ...     MobileViTConfig,
    ...     MobileViTImageClassify,
    ... )

    >>> configuration = MobileViTConfig()
    >>> model = MobileViTImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mobilevit"

    initial_dims: int = 16
    head_dims: int = 640
    block_dims: tuple = (32, 64, 96, 128, 160)
    expansion_ratio: tuple = (4.0, 4.0, 4.0, 4.0, 4.0)
    attention_dims: tuple = (None, None, 144, 192, 240)
    image_size: int = 256
    num_classes: int = 1000


# Hosted classification variants -> (model arch key, timm id). Weights load by Hub
# repo id (kf_config.json); the github release urls have been removed.
MOBILEVIT_VARIANTS = {
    "mobilevit_xxs_cvnets_in1k": {
        "model": "mobilevit_xxs",
        "timm_id": "mobilevit_xxs.cvnets_in1k",
    },
    "mobilevit_xs_cvnets_in1k": {
        "model": "mobilevit_xs",
        "timm_id": "mobilevit_xs.cvnets_in1k",
    },
    "mobilevit_s_cvnets_in1k": {
        "model": "mobilevit_s",
        "timm_id": "mobilevit_s.cvnets_in1k",
    },
}

MOBILEVIT_SEGMENT_MODEL_CONFIG = {
    "mobilevit_xxs_deeplabv3": {
        "initial_dims": 16,
        "head_dims": 320,
        "block_dims": [16, 24, 48, 64, 80],
        "expansion_ratio": [2.0, 2.0, 2.0, 2.0, 2.0],
        "attention_dims": [None, None, 64, 80, 96],
        "image_size": 512,
        "num_classes": 21,
    },
    "mobilevit_xs_deeplabv3": {
        "initial_dims": 16,
        "head_dims": 384,
        "block_dims": [32, 48, 64, 80, 96],
        "expansion_ratio": [4.0, 4.0, 4.0, 4.0, 4.0],
        "attention_dims": [None, None, 96, 120, 144],
        "image_size": 512,
        "num_classes": 21,
    },
    "mobilevit_s_deeplabv3": {
        "initial_dims": 16,
        "head_dims": 640,
        "block_dims": [32, 64, 96, 128, 160],
        "expansion_ratio": [4.0, 4.0, 4.0, 4.0, 4.0],
        "attention_dims": [None, None, 144, 192, 240],
        "image_size": 512,
        "num_classes": 21,
    },
}

MOBILEVIT_SEGMENT_WEIGHTS_URLS = {
    "mobilevit_xxs_deeplabv3": {
        "model": "mobilevit_xxs_deeplabv3",
        "url": "https://huggingface.co/kerasformers/mobilevit_xxs_deeplabv3",
    },
    "mobilevit_xs_deeplabv3": {
        "model": "mobilevit_xs_deeplabv3",
        "url": "https://huggingface.co/kerasformers/mobilevit_xs_deeplabv3",
    },
    "mobilevit_s_deeplabv3": {
        "model": "mobilevit_s_deeplabv3",
        "url": "https://huggingface.co/kerasformers/mobilevit_s_deeplabv3",
    },
}
