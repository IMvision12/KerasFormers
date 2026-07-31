from kerasformers.base import BaseConfig


class Owlv2Config(BaseConfig):
    r"""Configuration for [`Owlv2Detect`], the OWLv2 open-vocabulary detector.

    The defaults describe the owlv2-base-patch16 style. Other variants override
    the vision / text dimensions. Fields serialize flat to a repo's
    `kf_config.json`.

    Args:
        vision_image_size (`int`, *optional*, defaults to 960):
            Input image resolution of the vision tower.
        vision_patch_size (`int`, *optional*, defaults to 16):
            Patch size of the vision tower.
        vision_hidden_dim (`int`, *optional*, defaults to 768):
            Hidden dimension of the vision tower.
        vision_intermediate_size (`int`, *optional*, defaults to 3072):
            Feed-forward dimension of the vision tower.
        vision_num_layers (`int`, *optional*, defaults to 12):
            Number of transformer layers in the vision tower.
        vision_num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in the vision tower.
        text_hidden_dim (`int`, *optional*, defaults to 512):
            Hidden dimension of the text tower.
        text_intermediate_size (`int`, *optional*, defaults to 2048):
            Feed-forward dimension of the text tower.
        text_num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the text tower.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimension of the shared vision-text projection space.
        text_num_layers (`int`, *optional*, defaults to 12):
            Number of transformer layers in the text tower.
        text_max_position_embeddings (`int`, *optional*, defaults to 16):
            Maximum text sequence length (per prompt) the text tower handles.
        text_vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the CLIP text tokenizer.
        image_size (`int`, *optional*, defaults to `None`):
            Square input resolution to build for; `None` uses `vision_image_size`.

    Examples:

    ```python
    >>> from kerasformers.models.owlv2 import Owlv2Config, Owlv2Detect

    >>> configuration = Owlv2Config()
    >>> model = Owlv2Detect(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "owlv2"

    vision_image_size: int = 960
    vision_patch_size: int = 16
    vision_hidden_dim: int = 768
    vision_intermediate_size: int = 3072
    vision_num_layers: int = 12
    vision_num_heads: int = 12
    text_hidden_dim: int = 512
    text_intermediate_size: int = 2048
    text_num_heads: int = 8
    projection_dim: int = 512
    text_num_layers: int = 12
    text_max_position_embeddings: int = 16
    text_vocab_size: int = 49408
    image_size: int = None
