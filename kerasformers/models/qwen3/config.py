# Qwen3 — dense decoder-only LLM. Like Qwen2 but the attention adds per-head
# QK-norm (RMSNorm on q/k) and drops the q/k/v bias. RMSNorm, GQA, SwiGLU,
# 1D rotary. Loaded on the fly from Hugging Face:
#
#     Qwen3Generate.from_weights("hf:Qwen/Qwen3-0.6B")

QWEN3_CONFIG = {
    "qwen3-0.6b": {
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
    },
    "qwen3-1.7b": {
        "vocab_size": 151936,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
    },
    "qwen3-4b": {
        "vocab_size": 151936,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
    },
    "qwen3-8b": {
        "vocab_size": 151936,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": False,
    },
}
