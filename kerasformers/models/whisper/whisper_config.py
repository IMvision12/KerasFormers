from kerasformers.base import BaseConfig


class WhisperConfig(BaseConfig):
    r"""Configuration for the Whisper encoder-decoder ASR model.

    The defaults describe the Whisper Tiny variant; other variants override the
    encoder / decoder dimensions, vocabulary, and mel-bin count. One
    `kf_config.json` (declaring the canonical [`WhisperModel`]) sits on each
    variant's repo, and both [`WhisperModel`] and [`WhisperSpeechToText`] load from
    it. Fields mirror the model constructor and serialize flat.

    Args:
        hidden_dim (`int`, *optional*, defaults to 384):
            Hidden / embedding dimension.
        encoder_num_layers (`int`, *optional*, defaults to 4):
            Number of encoder transformer blocks.
        decoder_num_layers (`int`, *optional*, defaults to 4):
            Number of decoder transformer blocks.
        encoder_attention_heads (`int`, *optional*, defaults to 6):
            Encoder self-attention head count.
        decoder_attention_heads (`int`, *optional*, defaults to 6):
            Decoder self-attention / cross-attention head count.
        encoder_ffn_dim (`int`, *optional*, defaults to 1536):
            Encoder MLP hidden dimension.
        decoder_ffn_dim (`int`, *optional*, defaults to 1536):
            Decoder MLP hidden dimension.
        num_mel_bins (`int`, *optional*, defaults to 80):
            Mel-bin count of the input log-mel spectrogram (128 for large-v3).
        max_source_positions (`int`, *optional*, defaults to 1500):
            Maximum encoder position.
        max_target_positions (`int`, *optional*, defaults to 448):
            Maximum decoded length.
        vocab_size (`int`, *optional*, defaults to 51865):
            Token vocabulary size (51866 for the v3 variants).
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            MLP activation (`"gelu"` exact, matches OpenAI).
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for every LayerNorm.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Whether to scale the decoder token embedding by `sqrt(hidden_dim)`.

    Examples:

    ```python
    >>> from kerasformers.models.whisper import WhisperConfig, WhisperModel

    >>> configuration = WhisperConfig()
    >>> model = WhisperModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "whisper"

    hidden_dim: int = 384
    encoder_num_layers: int = 4
    decoder_num_layers: int = 4
    encoder_attention_heads: int = 6
    decoder_attention_heads: int = 6
    encoder_ffn_dim: int = 1536
    decoder_ffn_dim: int = 1536
    num_mel_bins: int = 80
    max_source_positions: int = 1500
    max_target_positions: int = 448
    vocab_size: int = 51865
    activation_function: str = "gelu"
    layer_norm_eps: float = 1e-5
    scale_embedding: bool = False
