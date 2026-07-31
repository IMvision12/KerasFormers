from kerasformers.base import BaseConfig


class MetaClip2Config(BaseConfig):
    r"""Configuration for the MetaCLIP 2 dual encoder ([`MetaClip2Model`] + heads).

    The defaults describe a MetaCLIP 2 Worldwide B/32 style model; other variants
    override the vision / text dimensions, patch size, and (for the mt5 variants)
    the vocabulary and `eos_token_id`. One `kf_config.json` (declaring the
    canonical [`MetaClip2ZeroShotClassify`]) sits on each variant's repo, and every
    MetaCLIP 2 head loads from it. Fields mirror the model constructor.

    Args:
        embed_dim (`int`, *optional*, defaults to 512):
            Shared image-text projection dimension.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the vision tower is built for.
        vision_num_layers (`int`, *optional*, defaults to 12):
            Depth of the ViT vision encoder.
        vision_hidden_dim (`int`, *optional*, defaults to 768):
            Hidden size of the vision encoder.
        vision_patch_size (`int`, *optional*, defaults to 32):
            Patch size of the vision encoder.
        vision_num_heads (`int`, *optional*, defaults to `None`):
            Number of attention heads in the vision encoder; when `None`, derived
            as `vision_hidden_dim // 64`.
        max_seq_len (`int`, *optional*, defaults to 77):
            Maximum text sequence length.
        vocab_size (`int`, *optional*, defaults to 901629):
            Text tokenizer vocabulary size (the multilingual XLM-R vocab; the mt5
            variants use 250000).
        text_hidden_dim (`int`, *optional*, defaults to 512):
            Hidden size of the text encoder.
        text_num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the text encoder.
        text_num_layers (`int`, *optional*, defaults to 12):
            Depth of the text encoder.
        vision_mlp_ratio (`float`, *optional*, defaults to 4.0):
            MLP expansion ratio in the vision blocks.
        text_mlp_ratio (`float`, *optional*, defaults to 4.0):
            MLP expansion ratio in the text blocks.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            MLP activation (`"quick_gelu"` for the huge-quickgelu variant).
        eos_token_id (`int`, *optional*, defaults to 2):
            End-of-sequence token id used to pool the text features (the mt5
            variants use 1).

    Examples:

    ```python
    >>> from kerasformers.models.metaclip2 import MetaClip2Config, MetaClip2Model

    >>> configuration = MetaClip2Config()
    >>> model = MetaClip2Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "metaclip_2"

    embed_dim: int = 512
    image_size: int = 224
    vision_num_layers: int = 12
    vision_hidden_dim: int = 768
    vision_patch_size: int = 32
    vision_num_heads: int = None
    max_seq_len: int = 77
    vocab_size: int = 901629
    text_hidden_dim: int = 512
    text_num_heads: int = 8
    text_num_layers: int = 12
    vision_mlp_ratio: float = 4.0
    text_mlp_ratio: float = 4.0
    hidden_act: str = "gelu"
    eos_token_id: int = 2
