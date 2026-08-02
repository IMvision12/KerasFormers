from kerasformers.base import BaseConfig


class DeepseekVLConfig(BaseConfig):
    r"""Configuration for the DeepSeek-VL backbone ([`DeepseekVLModel`]) and its
    generative head ([`DeepseekVLGenerate`]).

    DeepSeek-VL is a SigLIP vision tower + 2-linear GELU aligner + Llama decoder.
    One `kf_config.json` (declaring the canonical [`DeepseekVLModel`]) sits on each
    variant's repo; both the backbone and the generative head load from it. Fields
    mirror the model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 5632):
            SwiGLU hidden width per text layer.
        num_layers (`int`, *optional*, defaults to 24):
            Number of text decoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Query heads per text layer.
        num_kv_heads (`int`, *optional*, defaults to 16):
            Key/value heads per text layer.
        head_dim (`int`, *optional*, defaults to 128):
            Text per-head dim.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            Text RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Rotary base frequency.
        tie_embeddings (`bool`, *optional*, defaults to `False`):
            Whether [`DeepseekVLGenerate`] ties the LM head to the token embeddings.
        vision_embed_dim (`int`, *optional*, defaults to 1024):
            SigLIP vision tower hidden width.
        vision_mlp_dim (`int`, *optional*, defaults to 4096):
            SigLIP vision tower MLP width.
        vision_num_layers (`int`, *optional*, defaults to 24):
            Number of SigLIP encoder blocks.
        vision_num_heads (`int`, *optional*, defaults to 16):
            SigLIP attention heads.
        image_size (`int`, *optional*, defaults to 384):
            Square vision input size in pixels.
        patch_size (`int`, *optional*, defaults to 16):
            Vision patch size in pixels.
        vision_norm_eps (`float`, *optional*, defaults to 1e-6):
            Vision LayerNorm epsilon.
        image_token_id (`int`, *optional*, defaults to 100015):
            The `<image_placeholder>` token id whose slots receive image features.

    Examples:

    ```python
    >>> from kerasformers.models.deepseek_vl import DeepseekVLConfig, DeepseekVLModel

    >>> configuration = DeepseekVLConfig()
    >>> model = DeepseekVLModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_vl"

    vocab_size: int = 102400
    embed_dim: int = 2048
    mlp_dim: int = 5632
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int = 128
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_embeddings: bool = False
    vision_embed_dim: int = 1024
    vision_mlp_dim: int = 4096
    vision_num_layers: int = 24
    vision_num_heads: int = 16
    image_size: int = 384
    patch_size: int = 16
    vision_norm_eps: float = 1e-6
    image_token_id: int = 100015
