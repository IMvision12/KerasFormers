from kerasformers.base import BaseConfig


class CLIPConfig(BaseConfig):
    r"""Configuration for the CLIP dual encoder ([`CLIPModel`] and its task heads).

    The defaults describe an OpenAI CLIP ViT-B/32 style model; other variants
    override the vision / text dimensions and patch size. One `kf_config.json`
    (declaring the canonical [`CLIPModel`]) sits on each variant's repo, and the
    vision / text / embed / zero-shot / classify heads all load from it. Fields
    mirror the model constructor and serialize flat.

    Args:
        embed_dim (`int`, *optional*, defaults to 512):
            Shared image-text projection dimension (`projection_dim`).
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the vision tower is built for.
        vision_num_layers (`int`, *optional*, defaults to 12):
            Depth of the ViT vision encoder.
        vision_hidden_dim (`int`, *optional*, defaults to 768):
            Hidden size of the vision encoder.
        vision_patch_size (`int`, *optional*, defaults to 32):
            Patch size of the vision encoder.
        max_seq_len (`int`, *optional*, defaults to 77):
            Maximum text sequence length.
        vocab_size (`int`, *optional*, defaults to 49408):
            Text tokenizer vocabulary size.
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
        hidden_act (`str`, *optional*, defaults to `"quick_gelu"`):
            MLP activation: `"quick_gelu"` for canonical OpenAI CLIP, `"gelu"` for
            the larger LAION / open_clip variants.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for every LayerNorm.

    Examples:

    ```python
    >>> from kerasformers.models.clip import CLIPConfig, CLIPModel

    >>> configuration = CLIPConfig()
    >>> model = CLIPModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "clip"

    embed_dim: int = 512
    image_size: int = 224
    vision_num_layers: int = 12
    vision_hidden_dim: int = 768
    vision_patch_size: int = 32
    max_seq_len: int = 77
    vocab_size: int = 49408
    text_hidden_dim: int = 512
    text_num_heads: int = 8
    text_num_layers: int = 12
    vision_mlp_ratio: float = 4.0
    text_mlp_ratio: float = 4.0
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
