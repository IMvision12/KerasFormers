# Moonshine - Useful Sensors raw-waveform ASR, encoder-decoder.
# Architecture knobs shared by all variants: raw 16 kHz waveform input,
# 3x Conv1d (k=127/s=64 no-bias, k=7/s=3, k=3/s=2) + GroupNorm stem,
# rotary (GLM-style interleaved, partial) positions, no-bias LayerNorm,
# pre-norm blocks, gelu encoder FFN, gated-silu decoder FFN, tied LM head.

MOONSHINE_CONFIG = {
    "moonshine_tiny": {
        "hidden_dim": 288,
        "encoder_num_layers": 6,
        "decoder_num_layers": 6,
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 8,
        "encoder_ffn_dim": 1152,
        "decoder_ffn_dim": 1152,
        "vocab_size": 32768,
        "max_position_embeddings": 194,
        "partial_rotary_factor": 0.9,
        "rope_theta": 10000.0,
        "encoder_activation": "gelu",
        "decoder_activation": "silu",
    },
    "moonshine_base": {
        "hidden_dim": 416,
        "encoder_num_layers": 8,
        "decoder_num_layers": 8,
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 8,
        "encoder_ffn_dim": 1664,
        "decoder_ffn_dim": 1664,
        "vocab_size": 32768,
        "max_position_embeddings": 194,
        "partial_rotary_factor": 0.62,
        "rope_theta": 10000.0,
        "encoder_activation": "gelu",
        "decoder_activation": "silu",
    },
}


# Release weights: the converter's `__main__` saves `{variant}_usefulsensors.weights.h5`,
# which the user uploads to the GitHub `moonshine` release tag. On-the-fly conversion
# stays available via `from_weights("hf:UsefulSensors/moonshine-...")`.
MOONSHINE_WEIGHTS = {
    "moonshine_tiny": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/moonshine/moonshine_tiny_usefulsensors.weights.h5"
    },
    "moonshine_base": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/moonshine/moonshine_base_usefulsensors.weights.h5"
    },
}


MOONSHINE_HF_REPO = {
    "moonshine_tiny": "UsefulSensors/moonshine-tiny",
    "moonshine_base": "UsefulSensors/moonshine-base",
}
