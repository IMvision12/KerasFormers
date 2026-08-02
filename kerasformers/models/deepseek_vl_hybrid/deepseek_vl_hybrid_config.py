from kerasformers.base import BaseConfig


class DeepseekVLHybridConfig(BaseConfig):
    r"""Configuration for the DeepSeek-VL-Hybrid backbone
    ([`DeepseekVLHybridModel`]) and its generative head
    ([`DeepseekVLHybridGenerate`]).

    DeepSeek-VL Hybrid (7B) is a dual vision stack (SigLIP @384 + SAM/ViTDet
    @1024) + 3-way aligner + Llama-7B decoder. One `kf_config.json` (declaring the
    canonical [`DeepseekVLHybridModel`]) sits on each variant's repo; both the
    backbone and the generative head load from it. Fields mirror the model
    constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 4096):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 11008):
            SwiGLU hidden width per text layer.
        num_layers (`int`, *optional*, defaults to 30):
            Number of text decoder blocks.
        num_heads (`int`, *optional*, defaults to 32):
            Query heads per text layer.
        num_kv_heads (`int`, *optional*, defaults to 32):
            Key/value heads per text layer.
        head_dim (`int`, *optional*, defaults to 128):
            Text per-head dim.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            Text RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Rotary base frequency.
        tie_embeddings (`bool`, *optional*, defaults to `False`):
            Whether [`DeepseekVLHybridGenerate`] ties the LM head to the token
            embeddings.
        vision_embed_dim (`int`, *optional*, defaults to 1024):
            Low-res SigLIP vision tower hidden width.
        vision_mlp_dim (`int`, *optional*, defaults to 4096):
            Low-res SigLIP vision tower MLP width.
        vision_num_layers (`int`, *optional*, defaults to 24):
            Number of low-res SigLIP encoder blocks.
        vision_num_heads (`int`, *optional*, defaults to 16):
            Low-res SigLIP attention heads.
        image_size (`int`, *optional*, defaults to 384):
            Square low-res (SigLIP) input size in pixels.
        patch_size (`int`, *optional*, defaults to 16):
            Low-res vision patch size in pixels.
        vision_norm_eps (`float`, *optional*, defaults to 1e-6):
            Low-res vision LayerNorm epsilon.
        high_res_embed_dim (`int`, *optional*, defaults to 768):
            High-res SAM/ViTDet tower hidden width.
        high_res_mlp_dim (`int`, *optional*, defaults to 3072):
            High-res SAM/ViTDet tower MLP width.
        high_res_num_layers (`int`, *optional*, defaults to 12):
            Number of high-res SAM/ViTDet encoder blocks.
        high_res_num_heads (`int`, *optional*, defaults to 12):
            High-res SAM/ViTDet attention heads.
        high_res_image_size (`int`, *optional*, defaults to 1024):
            Square high-res (SAM) input size in pixels.
        high_res_patch_size (`int`, *optional*, defaults to 16):
            High-res vision patch size in pixels.
        high_res_output_channels (`int`, *optional*, defaults to 256):
            SAM neck output channel count.
        high_res_window_size (`int`, *optional*, defaults to 14):
            SAM windowed-attention window size.
        high_res_global_attn_indexes (`tuple`, *optional*, defaults to `(2, 5, 8, 11)`):
            Indices of the SAM blocks that use global (non-windowed) attention.
        high_res_norm_eps (`float`, *optional*, defaults to 1e-6):
            High-res vision LayerNorm epsilon.
        image_token_id (`int`, *optional*, defaults to 100015):
            The `<image_placeholder>` token id whose slots receive image features.

    Examples:

    ```python
    >>> from kerasformers.models.deepseek_vl_hybrid import (
    ...     DeepseekVLHybridConfig,
    ...     DeepseekVLHybridModel,
    ... )

    >>> configuration = DeepseekVLHybridConfig()
    >>> model = DeepseekVLHybridModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_vl_hybrid"

    vocab_size: int = 102400
    embed_dim: int = 4096
    mlp_dim: int = 11008
    num_layers: int = 30
    num_heads: int = 32
    num_kv_heads: int = 32
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
    high_res_embed_dim: int = 768
    high_res_mlp_dim: int = 3072
    high_res_num_layers: int = 12
    high_res_num_heads: int = 12
    high_res_image_size: int = 1024
    high_res_patch_size: int = 16
    high_res_output_channels: int = 256
    high_res_window_size: int = 14
    high_res_global_attn_indexes: tuple = (2, 5, 8, 11)
    high_res_norm_eps: float = 1e-6
    image_token_id: int = 100015
