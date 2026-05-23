# Qwen2 — Alibaba's dense decoder-only LLM (RMSNorm, GQA with q/k/v bias,
# SwiGLU, 1D rotary positions). Loaded on the fly from Hugging Face:
#
#     Qwen2Generate.from_weights("hf:Qwen/Qwen2-0.5B-Instruct")
#
# (Qwen2.5 checkpoints share this architecture — `model_type == "qwen2"` — and
# load through the same class.)

QWEN2_CONFIG = {
    "qwen2-0.5b-instruct": {
        "vocab_size": 151936,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
    },
    "qwen2-1.5b-instruct": {
        "vocab_size": 151936,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
    },
    "qwen2-7b-instruct": {
        "vocab_size": 152064,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": False,
    },
}
