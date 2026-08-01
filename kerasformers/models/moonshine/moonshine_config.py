from kerasformers.base import BaseConfig


class MoonshineConfig(BaseConfig):
    r"""Configuration for the Moonshine encoder-decoder ASR model.

    Moonshine runs on raw waveform (no mel spectrogram) with partial rotary
    positions. The defaults describe the Moonshine Tiny variant; Base overrides the
    dimensions and `partial_rotary_factor`. One `kf_config.json` (declaring the
    canonical [`MoonshineModel`]) sits on each variant's repo, and both
    [`MoonshineModel`] and [`MoonshineSpeechToText`] load from it. Fields mirror the
    model constructor and serialize flat.

    Args:
        hidden_dim (`int`, *optional*, defaults to 288):
            Hidden / embedding dimension.
        encoder_num_layers (`int`, *optional*, defaults to 6):
            Number of encoder transformer blocks.
        decoder_num_layers (`int`, *optional*, defaults to 6):
            Number of decoder transformer blocks.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Encoder self-attention head count.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Decoder self-attention / cross-attention head count.
        encoder_num_kv_heads (`int`, *optional*, defaults to `None`):
            Encoder key/value head count; when `None`, equals the attention heads.
        decoder_num_kv_heads (`int`, *optional*, defaults to `None`):
            Decoder key/value head count; when `None`, equals the attention heads.
        encoder_ffn_dim (`int`, *optional*, defaults to 1152):
            Encoder MLP hidden dimension.
        decoder_ffn_dim (`int`, *optional*, defaults to 1152):
            Decoder MLP hidden dimension.
        vocab_size (`int`, *optional*, defaults to 32768):
            Token vocabulary size.
        max_position_embeddings (`int`, *optional*, defaults to 194):
            Maximum sequence length the rotary cache is built for.
        partial_rotary_factor (`float`, *optional*, defaults to 0.9):
            Fraction of head dimensions that receive rotary embedding (0.62 Base).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Rotary-embedding base frequency.
        encoder_activation (`str`, *optional*, defaults to `"gelu"`):
            Encoder MLP activation.
        decoder_activation (`str`, *optional*, defaults to `"silu"`):
            Decoder MLP activation (gated SiLU).
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for every LayerNorm.

    Examples:

    ```python
    >>> from kerasformers.models.moonshine import MoonshineConfig, MoonshineModel

    >>> configuration = MoonshineConfig()
    >>> model = MoonshineModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "moonshine"

    hidden_dim: int = 288
    encoder_num_layers: int = 6
    decoder_num_layers: int = 6
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_num_kv_heads: int = None
    decoder_num_kv_heads: int = None
    encoder_ffn_dim: int = 1152
    decoder_ffn_dim: int = 1152
    vocab_size: int = 32768
    max_position_embeddings: int = 194
    partial_rotary_factor: float = 0.9
    rope_theta: float = 10000.0
    encoder_activation: str = "gelu"
    decoder_activation: str = "silu"
    layer_norm_eps: float = 1e-5
