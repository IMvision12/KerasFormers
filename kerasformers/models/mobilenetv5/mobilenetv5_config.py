from kerasformers.base import BaseConfig


class MobileNetV5Config(BaseConfig):
    r"""Configuration for [`MobileNetV5Encoder`].

    MobileNetV5 is the vision encoder introduced with Gemma 3n. It reuses the
    MobileNetV4 block vocabulary (EdgeResidual, Universal Inverted Bottleneck, Mobile
    Multi-Query Attention) but swaps BatchNorm for RmsNorm2d, uses tanh-approximate
    GELU, TF-style ``same`` padding, and terminates in a Multi-Scale Fusion Adapter
    (MSFA) that fuses the last two stage feature maps into a single ``(H, W, 2048)``
    token grid instead of a classifier head. This is a standalone port from timm's
    ``mobilenetv5_300m_enc`` (independent of the Gemma 3n vision tower).

    Args:
        config (`str`, *optional*, defaults to `"300m"`):
            Variant key selecting the block schedule. One of `"300m"`, `"base"`.
        image_size (`int`, *optional*, defaults to 768):
            Square input resolution the encoder is built for.
        msfa_output_resolution (`int`, *optional*, defaults to 16):
            Spatial resolution of the MSFA output token grid.

    Examples:

    ```python
    >>> from kerasformers.models.mobilenetv5 import (
    ...     MobileNetV5Config,
    ...     MobileNetV5Encoder,
    ... )

    >>> configuration = MobileNetV5Config()
    >>> model = MobileNetV5Encoder(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mobilenetv5"

    config: str = "300m"
    image_size: int = 768
    msfa_output_resolution: int = 16
