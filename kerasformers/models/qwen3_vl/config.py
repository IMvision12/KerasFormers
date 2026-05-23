# Qwen3-VL — third-gen Qwen multimodal LLM. Bigger architectural deltas:
#   * Text decoder is Qwen3 (QK-norm in attention, no qkv bias), rope_theta 5e6,
#     and INTERLEAVED M-RoPE (mrope_interleaved) instead of sectioned.
#   * Vision tower: Conv3d (with bias) patch embed (patch_size 16), bilinearly
#     interpolated LEARNED position embeddings, LayerNorm + (non-gated) GELU MLP
#     blocks, full attention (no windowing), and DeepStack — features from a few
#     vision layers are injected into the first text layers.
#
# Loaded on the fly from HF (no kerasformers release uploads):
#     Qwen3VLModel.from_weights("hf:Qwen/Qwen3-VL-2B-Instruct")

QWEN3_VL_CONFIG = {
    "qwen3-vl-2b-instruct": {
        "vocab_size": 151936,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000.0,
        "mrope_section": (24, 20, 20),
        "tie_word_embeddings": True,
        "vision_depth": 24,
        "vision_hidden_size": 1024,
        "vision_intermediate_size": 4096,
        "vision_num_heads": 16,
        "vision_out_hidden_size": 2048,
        "vision_hidden_act": "gelu_pytorch_tanh",
        "num_position_embeddings": 2304,
        "deepstack_visual_indexes": (5, 11, 17),
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}

QWEN3_VL_TOKENS = {
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
}
