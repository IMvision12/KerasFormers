from kerasformers.base import BaseConfig


class Speech2TextConfig(BaseConfig):
    r"""Configuration for the Speech2Text (S2T) encoder-decoder ASR model.

    The defaults describe the S2T Small LibriSpeech variant; other variants
    override the encoder / decoder dimensions and head counts. One
    `kf_config.json` (declaring the canonical [`Speech2TextModel`]) sits on each
    variant's repo, and both [`Speech2TextModel`] and [`Speech2TextSpeechToText`]
    load from it. Fields mirror the model constructor and serialize flat.

    Args:
        hidden_dim (`int`, *optional*, defaults to 256):
            Hidden / embedding dimension.
        encoder_num_layers (`int`, *optional*, defaults to 12):
            Number of encoder transformer blocks.
        decoder_num_layers (`int`, *optional*, defaults to 6):
            Number of decoder transformer blocks.
        encoder_attention_heads (`int`, *optional*, defaults to 4):
            Encoder self-attention head count.
        decoder_attention_heads (`int`, *optional*, defaults to 4):
            Decoder self-attention / cross-attention head count.
        encoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Encoder MLP hidden dimension.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Decoder MLP hidden dimension.
        vocab_size (`int`, *optional*, defaults to 10000):
            SentencePiece token vocabulary size.
        num_mel_bins (`int`, *optional*, defaults to 80):
            Mel-filterbank channel count of the input features.
        max_source_positions (`int`, *optional*, defaults to 6000):
            Maximum encoder position.
        max_target_positions (`int`, *optional*, defaults to 1024):
            Maximum decoded length.
        conv_channels (`int`, *optional*, defaults to 1024):
            Channel count of the Conv1d subsampler.
        conv_kernel_sizes (`tuple`, *optional*, defaults to `(5, 5)`):
            Kernel sizes of the Conv1d subsampler layers.
        num_conv_layers (`int`, *optional*, defaults to 2):
            Number of Conv1d subsampler layers.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Whether to scale the token embedding by `sqrt(hidden_dim)`.
        activation_function (`str`, *optional*, defaults to `"relu"`):
            MLP activation.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for every LayerNorm.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id (used by the decoder position embedding).

    Examples:

    ```python
    >>> from kerasformers.models.speech2text import Speech2TextConfig, Speech2TextModel

    >>> configuration = Speech2TextConfig()
    >>> model = Speech2TextModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "speech_to_text"

    hidden_dim: int = 256
    encoder_num_layers: int = 12
    decoder_num_layers: int = 6
    encoder_attention_heads: int = 4
    decoder_attention_heads: int = 4
    encoder_ffn_dim: int = 2048
    decoder_ffn_dim: int = 2048
    vocab_size: int = 10000
    num_mel_bins: int = 80
    max_source_positions: int = 6000
    max_target_positions: int = 1024
    conv_channels: int = 1024
    conv_kernel_sizes: tuple = (5, 5)
    num_conv_layers: int = 2
    scale_embedding: bool = True
    activation_function: str = "relu"
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 1
