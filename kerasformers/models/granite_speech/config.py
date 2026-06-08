GRANITE_SPEECH_CONFIG = {
    "granite_speech_3_3_2b": {
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
        "cat_hidden_layers": None,
    },
}


GRANITE_SPEECH_WEIGHTS = {
    "granite_speech_3_3_2b": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/granite_speech/granite_speech_3_3_2b.weights.json",
    },
}


GRANITE_SPEECH_VOCAB_URL = "https://github.com/IMvision12/KerasFormers/releases/download/granite_speech/granite_speech_vocab.json"
GRANITE_SPEECH_MERGES_URL = "https://github.com/IMvision12/KerasFormers/releases/download/granite_speech/granite_speech_merges.txt"

# Granite-3.3 special tokens in id order. The first 19 live in vocab.json; the
# last 8 (start_of_role ... audio) are appended on top, so <|audio|> lands at its
# checkpoint id 49159. Embedded because vocab.json/merges.txt don't carry them.
GRANITE_SPEECH_SPECIAL_TOKENS = [
    "<|end_of_text|>",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
    "<filename>",
    "<gh_stars>",
    "<issue_start>",
    "<issue_comment>",
    "<issue_closed>",
    "<jupyter_start>",
    "<jupyter_text>",
    "<jupyter_code>",
    "<jupyter_output>",
    "<empty_output>",
    "<commit_before>",
    "<commit_msg>",
    "<commit_after>",
    "<reponame>",
    "<|start_of_role|>",
    "<|end_of_role|>",
    "<|tool_call|>",
    "<|start_of_cite|>",
    "<|end_of_cite|>",
    "<|start_of_plugin|>",
    "<|end_of_plugin|>",
    "<|audio|>",
]
