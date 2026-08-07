from kerasformers.base import BaseConfig


class Gemma3Config(BaseConfig):
    r"""Configuration for Gemma 3: [`Gemma3Model`] and [`Gemma3Generate`].

    Gemma 3's decoder recipe (scaled embeddings, `(1 + w)` RMSNorms, per-head QK
    norms, the four-norm sandwich, a 5:1 sliding-to-global layer pattern, and dual
    rotary bases: `rope_local_theta` on sliding layers, `rope_theta` with an optional
    linear `rope_scaling_factor` on global layers), plus the optional SigLIP vision
    tower + average-pool projector. `vision_num_layers=0` builds the text-only 1B; the
    4B / 12B / 27B set the `vision_*` and `mm_*` fields for image + text. Fields
    serialize flat into each repo's one `kf_config.json` and mirror the model
    constructor.

    Args:
        vocab_size (`int`, *optional*, defaults to 262144):
            Token vocabulary size (262208 on the multimodal sizes).
        embed_dim (`int`, *optional*, defaults to 1152):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 6912):
            GeGLU hidden width per layer.
        num_layers (`int`, *optional*, defaults to 26):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 4):
            Query heads per layer.
        num_kv_heads (`int`, *optional*, defaults to 1):
            Key/value heads per layer.
        head_dim (`int`, *optional*, defaults to 256):
            Per-head dimension.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256.0):
            Attention scaling denominator (168.0 on the 27B).
        sliding_window (`int`, *optional*, defaults to 512):
            Window of the sliding-attention layers (1024 on the multimodal sizes).
        sliding_window_pattern (`int`, *optional*, defaults to 6):
            Every `pattern`-th layer is global.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Global-layer rotary base.
        rope_local_theta (`float`, *optional*, defaults to 10000.0):
            Sliding-layer rotary base.
        rope_scaling_factor (`float`, *optional*):
            Linear factor dividing the global-layer inverse frequencies (`None`
            disables; 8.0 on the multimodal sizes).
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether [`Gemma3Generate`] ties the LM head to the token embedding.
        vision_embed_dim (`int`, *optional*, defaults to 1152):
            SigLIP tower width.
        vision_mlp_dim (`int`, *optional*, defaults to 4304):
            SigLIP MLP width.
        vision_num_layers (`int`, *optional*, defaults to 0):
            SigLIP encoder blocks (0 builds the text-only 1B; 27 on 4B+).
        vision_num_heads (`int`, *optional*, defaults to 16):
            SigLIP attention heads.
        image_size (`int`, *optional*, defaults to 896):
            Vision input side length.
        patch_size (`int`, *optional*, defaults to 14):
            Vision patch side length.
        vision_norm_eps (`float`, *optional*, defaults to 1e-6):
            Vision LayerNorm epsilon.
        mm_tokens_per_image (`int`, *optional*, defaults to 256):
            Image tokens after pooling.
        image_token_id (`int`, *optional*, defaults to 262144):
            `<image_soft_token>` placeholder id.

    Examples:

    ```python
    >>> from kerasformers.models.gemma3 import Gemma3Config, Gemma3Generate

    >>> configuration = Gemma3Config(embed_dim=2560, num_layers=34, vision_num_layers=27)
    >>> model = Gemma3Generate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "gemma3"

    vocab_size: int = 262144
    embed_dim: int = 1152
    mlp_dim: int = 6912
    num_layers: int = 26
    num_heads: int = 4
    num_kv_heads: int = 1
    head_dim: int = 256
    query_pre_attn_scalar: float = 256.0
    sliding_window: int = 512
    sliding_window_pattern: int = 6
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_local_theta: float = 10000.0
    rope_scaling_factor: float | None = None
    tie_embeddings: bool = True
    vision_embed_dim: int = 1152
    vision_mlp_dim: int = 4304
    vision_num_layers: int = 0
    vision_num_heads: int = 16
    image_size: int = 896
    patch_size: int = 14
    vision_norm_eps: float = 1e-6
    mm_tokens_per_image: int = 256
    image_token_id: int = 262144
