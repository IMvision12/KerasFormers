# Speech2Text (S2T) - fairseq/Facebook conv-Transformer ASR, encoder-decoder.
# Architecture knobs shared by all LibriSpeech variants: 80-dim fbank input,
# 2x Conv1d (k=5, s=2) + GLU subsampler, sinusoidal positions, scaled
# embeddings, ReLU FFNs, SentencePiece (10k) vocabulary.

SPEECH2TEXT_CONFIG = {
    "s2t-small-librispeech-asr": {
        "hidden_dim": 256,
        "encoder_num_layers": 12,
        "decoder_num_layers": 6,
        "encoder_attention_heads": 4,
        "decoder_attention_heads": 4,
        "encoder_ffn_dim": 2048,
        "decoder_ffn_dim": 2048,
        "vocab_size": 10000,
        "num_mel_bins": 80,
        "max_source_positions": 6000,
        "max_target_positions": 1024,
        "conv_channels": 1024,
        "conv_kernel_sizes": (5, 5),
        "num_conv_layers": 2,
        "scale_embedding": True,
        "activation_function": "relu",
    },
    "s2t-medium-librispeech-asr": {
        "hidden_dim": 512,
        "encoder_num_layers": 12,
        "decoder_num_layers": 6,
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 8,
        "encoder_ffn_dim": 2048,
        "decoder_ffn_dim": 2048,
        "vocab_size": 10000,
        "num_mel_bins": 80,
        "max_source_positions": 6000,
        "max_target_positions": 1024,
        "conv_channels": 1024,
        "conv_kernel_sizes": (5, 5),
        "num_conv_layers": 2,
        "scale_embedding": True,
        "activation_function": "relu",
    },
    "s2t-large-librispeech-asr": {
        "hidden_dim": 1024,
        "encoder_num_layers": 12,
        "decoder_num_layers": 6,
        "encoder_attention_heads": 16,
        "decoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "decoder_ffn_dim": 4096,
        "vocab_size": 10000,
        "num_mel_bins": 80,
        "max_source_positions": 6000,
        "max_target_positions": 1024,
        "conv_channels": 1024,
        "conv_kernel_sizes": (5, 5),
        "num_conv_layers": 2,
        "scale_embedding": True,
        "activation_function": "relu",
    },
}


SPEECH2TEXT_WEIGHTS_URLS = {
    "s2t-small-librispeech-asr": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/speech2text/s2t_small_librispeech_asr.weights.h5",
    },
    "s2t-medium-librispeech-asr": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/speech2text/s2t_medium_librispeech_asr.weights.h5",
    },
    "s2t-large-librispeech-asr": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/speech2text/s2t_large_librispeech_asr.weights.h5",
    },
}


SPEECH2TEXT_TOKENIZER_FILES = {
    "vocab": "https://github.com/IMvision12/KerasFormers/releases/download/speech2text/s2t_vocab.json",
    "spm": "https://github.com/IMvision12/KerasFormers/releases/download/speech2text/s2t_spm.model",
}
