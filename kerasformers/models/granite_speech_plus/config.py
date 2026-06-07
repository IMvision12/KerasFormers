GRANITE_SPEECH_PLUS_CONFIG = {
    "granite_speech_4_1_2b_plus": {
        "vocab_size": 100353,
        "embed_dim": 2048,
        "mlp_dim": 4096,
        "num_layers": 40,
        "num_heads": 16,
        "num_kv_heads": 4,
        "norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22,
        "attention_multiplier": 0.0078125,
        "logits_scaling": 8.0,
        "tie_embeddings": True,
        "audio_token_id": 100352,
        "downsample_rate": 5,
        "window_size": 15,
        "has_lora_adapter": False,
        "lora_rank": 64,
        "lora_alpha": 32,
        "encoder_input_dim": 160,
        "encoder_num_layers": 16,
        "encoder_hidden_dim": 1024,
        "encoder_feedforward_mult": 4,
        "encoder_num_heads": 8,
        "encoder_dim_head": 128,
        "encoder_output_dim": 348,
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
        "cat_hidden_layers": [3],
    },
}


GRANITE_SPEECH_PLUS_WEIGHTS = {
    "granite_speech_4_1_2b_plus": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/granite_speech/granite_speech_4_1_2b_plus.weights.json",
    },
}


GRANITE_SPEECH_PLUS_TOKENIZER_URL = "https://github.com/IMvision12/KerasFormers/releases/download/granite_speech/granite_speech_plus_tokenizer.json"
