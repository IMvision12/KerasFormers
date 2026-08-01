from kerasformers.base import BaseConfig


class GraniteSpeechConfig(BaseConfig):
    r"""Configuration for Granite Speech ([`GraniteSpeechModel`] / [`GraniteSpeechGenerate`]).

    A conformer CTC audio encoder + BLIP-2 Q-Former projector feeding a Granite text
    decoder, fused at the `audio_token_id` placeholder positions. The defaults
    describe the Granite Speech 3.3 2B variant; other variants override the decoder
    dimensions, vocabulary, multipliers, and (for Plus) `cat_hidden_layers`. One
    `kf_config.json` (declaring the canonical [`GraniteSpeechGenerate`]) sits on each
    variant's repo, and both the backbone and the generate head load from it. Fields
    mirror the model constructor and serialize flat.

    Text decoder: `vocab_size`, `embed_dim`, `mlp_dim`, `num_layers`, `num_heads`,
    `num_kv_heads`, `norm_eps`, `rope_theta`, the Granite scalar multipliers
    (`embedding_multiplier`, `residual_multiplier`, `attention_multiplier`,
    `logits_scaling`), `tie_embeddings`, `eos_token_id`, `audio_token_id`.
    Audio fusion / LoRA: `downsample_rate`, `window_size`, `has_lora_adapter`,
    `lora_rank`, `lora_alpha`.
    Conformer encoder: the `encoder_*` fields. Q-Former projector: the `projector_*`
    fields. `cat_hidden_layers` (Plus only) lists the intermediate encoder layers
    concatenated with the final output before the projector.

    Examples:

    ```python
    >>> from kerasformers.models.granite_speech import GraniteSpeechConfig, GraniteSpeechGenerate

    >>> configuration = GraniteSpeechConfig()
    >>> model = GraniteSpeechGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "granite_speech"

    vocab_size: int = 49160
    embed_dim: int = 2048
    mlp_dim: int = 8192
    num_layers: int = 40
    num_heads: int = 32
    num_kv_heads: int = 8
    norm_eps: float = 1e-5
    rope_theta: float = 10000000.0
    embedding_multiplier: float = 12.0
    residual_multiplier: float = 0.22
    attention_multiplier: float = 0.015625
    logits_scaling: float = 8.0
    tie_embeddings: bool = True
    eos_token_id: int = 0
    audio_token_id: int = 49159
    downsample_rate: int = 5
    window_size: int = 15
    has_lora_adapter: bool = True
    lora_rank: int = 64
    lora_alpha: int = 32
    encoder_input_dim: int = 160
    encoder_num_layers: int = 16
    encoder_hidden_dim: int = 1024
    encoder_feedforward_mult: int = 4
    encoder_num_heads: int = 8
    encoder_dim_head: int = 128
    encoder_output_dim: int = 256
    encoder_context_size: int = 200
    encoder_max_pos_emb: int = 512
    encoder_conv_kernel_size: int = 15
    encoder_conv_expansion_factor: int = 2
    projector_dim: int = 1024
    projector_num_layers: int = 2
    projector_num_heads: int = 16
    projector_intermediate_size: int = 4096
    projector_cross_attention_frequency: int = 1
    projector_layer_norm_eps: float = 1e-12
    cat_hidden_layers: tuple = None
