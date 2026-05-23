# Qwen3.5 (text backbone) — the Qwen3-Next hybrid LLM. Most layers are
# Gated-DeltaNet linear attention (depthwise conv1d + delta-rule recurrence +
# gated RMSNorm); every `full_attention_interval`-th layer is gated full
# attention (QK-norm, partial rotary, output gate). Zero-centered (1+weight)
# RMSNorm throughout.
#
# Qwen3.5 ships as a multimodal series (Qwen3_5ForConditionalGeneration); this
# is the text decoder (model_type "qwen3_5_text"). Weights load on the fly from
# any Qwen3.5 checkpoint's `model.language_model.*` tensors:
#
#     Qwen3_5Generate.from_weights("hf:Qwen/Qwen3.5-0.8B")
#
# For the 0.8B linear_num_key_heads == linear_num_value_heads (16); larger
# variants use a 16/32 key/value-head split.

QWEN3_5_CONFIG = {
    "qwen3.5-0.8b": {
        "vocab_size": 248320,
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_word_embeddings": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
    },
    "qwen3.5-27b": {
        "vocab_size": 248320,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_word_embeddings": False,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
    },
}
