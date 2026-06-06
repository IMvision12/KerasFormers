# GraniteSpeechPlus: same architecture as GraniteSpeech, except the conformer CTC
# encoder concatenates a subset of intermediate layer outputs (`cat_hidden_layers`)
# with its final output before the projector, so the projector's encoder_hidden_size
# becomes `encoder_hidden_dim * (len(cat_hidden_layers) + 1)`. The model + layers are
# reused from `kerasformers.models.granite_speech`; only the config differs.

GRANITE_SPEECH_PLUS_CONFIG = {
    "granite-speech-4.1-2b-plus": {
        "vocab_size": 49160,
        "embed_dim": 2048,
        "mlp_dim": 8192,
        "num_layers": 40,
        "num_heads": 32,
        "num_kv_heads": 8,
        "norm_eps": 1e-5,
        "rope_theta": 10000000.0,
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22,
        "attention_multiplier": 0.015625,
        "logits_scaling": 8.0,
        "tie_embeddings": True,
        "audio_token_id": 49159,
        "downsample_rate": 5,
        "window_size": 15,
        "has_lora_adapter": True,
        "lora_rank": 64,
        "lora_alpha": 32,
        "encoder_input_dim": 160,
        "encoder_num_layers": 16,
        "encoder_hidden_dim": 1024,
        "encoder_feedforward_mult": 4,
        "encoder_num_heads": 8,
        "encoder_dim_head": 128,
        "encoder_output_dim": 256,
        "encoder_context_size": 200,
        "encoder_max_pos_emb": 512,
        "encoder_conv_kernel_size": 15,
        "encoder_conv_expansion_factor": 2,
        "projector_hidden_size": 1024,
        "projector_num_layers": 2,
        "projector_num_heads": 16,
        "projector_intermediate_size": 4096,
        "projector_cross_attention_frequency": 1,
        "projector_layer_norm_eps": 1e-12,
        # Placeholder index set — confirm against the real granite-speech-4.1-2b-plus
        # config.json (the model auto-sizes the projector from len(cat_hidden_layers)).
        "cat_hidden_layers": [7],
    },
}


GRANITE_SPEECH_PLUS_WEIGHTS = {
    "granite-speech-4.1-2b-plus": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/granite_speech/granite_speech_4_1_2b_plus.weights.json",
    },
}


GRANITE_SPEECH_PLUS_HF_IDS = {
    "granite-speech-4.1-2b-plus": "ibm-granite/granite-speech-4.1-2b-plus",
}
