from kerasformers.base import BaseConfig


class LocateAnythingConfig(BaseConfig):
    r"""Configuration for [`LocateAnythingGenerate`], LocateAnything-3B.

    A MoonViT vision encoder projected into a Qwen2.5-3B decoder, with Parallel
    Box Decoding for grounding. There is a single released variant; the defaults
    describe it. Fields mirror the model constructor and serialize flat to a
    repo's `kf_config.json`.

    Args:
        vocab_size (`int`, *optional*, defaults to 152681):
            Text tokenizer vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Hidden size of the Qwen2 decoder.
        mlp_dim (`int`, *optional*, defaults to 11008):
            Feed-forward dimension of the decoder.
        num_layers (`int`, *optional*, defaults to 36):
            Number of decoder layers.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        num_kv_heads (`int`, *optional*, defaults to 2):
            Number of key/value heads (grouped-query attention).
        head_dim (`int`, *optional*, defaults to 128):
            Per-head dimension.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Decoder rotary-embedding base frequency.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the LM head is tied to the token embedding.
        vision_embed_dim (`int`, *optional*, defaults to 1152):
            Hidden size of the MoonViT vision encoder.
        vision_depth (`int`, *optional*, defaults to 27):
            Number of vision encoder layers.
        vision_num_heads (`int`, *optional*, defaults to 16):
            Number of vision attention heads.
        vision_mlp_dim (`int`, *optional*, defaults to 4304):
            Feed-forward dimension of the vision encoder.
        vision_patch_size (`int`, *optional*, defaults to 14):
            Vision patch size.
        vision_init_pos_h (`int`, *optional*, defaults to 64):
            Height of the pretrained position-embedding grid (interpolated).
        vision_init_pos_w (`int`, *optional*, defaults to 64):
            Width of the pretrained position-embedding grid (interpolated).
        merge_kernel (`tuple`, *optional*, defaults to `(2, 2)`):
            Spatial merge kernel applied to vision tokens before projection.
        vision_rope_theta (`float`, *optional*, defaults to 10000.0):
            Vision rotary-embedding base frequency.
        image_token_index (`int`, *optional*, defaults to 151665):
            Token id whose positions are replaced by vision features.
        block_size (`int`, *optional*, defaults to 6):
            Parallel Box Decoding block size.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length the rotary cache is built for.

    Examples:

    ```python
    >>> from kerasformers.models.locateanything import (
    ...     LocateAnythingConfig, LocateAnythingGenerate)

    >>> configuration = LocateAnythingConfig()
    >>> model = LocateAnythingGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "locateanything"

    vocab_size: int = 152681
    embed_dim: int = 2048
    mlp_dim: int = 11008
    num_layers: int = 36
    num_heads: int = 16
    num_kv_heads: int = 2
    head_dim: int = 128
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_embeddings: bool = True
    vision_embed_dim: int = 1152
    vision_depth: int = 27
    vision_num_heads: int = 16
    vision_mlp_dim: int = 4304
    vision_patch_size: int = 14
    vision_init_pos_h: int = 64
    vision_init_pos_w: int = 64
    merge_kernel: tuple = (2, 2)
    vision_rope_theta: float = 10000.0
    image_token_index: int = 151665
    block_size: int = 6
    max_position_embeddings: int = 32768
