# Qwen2-VL — Alibaba's image+video+text multimodal LLM (vision encoder +
# Qwen2 decoder fused by M-RoPE). Architecture knobs shared by all variants:
# a ViT vision tower (Conv3d patch embed, 2D rotary, 2x2 patch merger) feeding
# a Qwen2 causal LLM (RMSNorm, GQA, SwiGLU, multimodal rotary positions).
#
# Weights are converted on the fly from the Hugging Face checkpoints — there
# are no kerasformers release uploads for this family, so there is no
# BASE_WEIGHT_CONFIG. Load with:
#
#     Qwen2VLModel.from_weights("hf:Qwen/Qwen2-VL-2B-Instruct")
#
# The variant entries below only carry architecture hyperparameters (so a bare
# `from_weights("qwen2-vl-2b-instruct", load_weights=False)` can size a model
# offline); the canonical path is the `hf:` one above.

QWEN2_VL_CONFIG = {
    "qwen2-vl-2b-instruct": {
        "vocab_size": 151936,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "mrope_section": (16, 24, 24),
        "tie_word_embeddings": True,
        "vision_depth": 32,
        "vision_embed_dim": 1280,
        "vision_num_heads": 16,
        "vision_mlp_ratio": 4,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "qwen2-vl-7b-instruct": {
        "vocab_size": 152064,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "mrope_section": (16, 24, 24),
        "tie_word_embeddings": False,
        "vision_depth": 32,
        "vision_embed_dim": 1280,
        "vision_num_heads": 16,
        "vision_mlp_ratio": 4,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "qwen2-vl-72b-instruct": {
        "vocab_size": 152064,
        "hidden_size": 8192,
        "intermediate_size": 29568,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "mrope_section": (16, 24, 24),
        "tie_word_embeddings": False,
        "vision_depth": 32,
        "vision_embed_dim": 1280,
        "vision_num_heads": 16,
        "vision_mlp_ratio": 4,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}

# Special token ids are identical across the Qwen2-VL variants.
QWEN2_VL_TOKENS = {
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
}
