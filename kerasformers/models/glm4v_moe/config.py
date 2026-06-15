# GLM-4.5V (single released checkpoint); GLM4V_MOE_COMMON holds the shared
# config (MoE decoder + vision tower + multimodal token ids) and the variant
# reuses a copy of it.
GLM4V_MOE_COMMON = {
    "vocab_size": 151424,
    "embed_dim": 4096,
    "mlp_dim": 10944,
    "moe_mlp_dim": 1408,
    "num_layers": 46,
    "num_heads": 96,
    "num_kv_heads": 8,
    "head_dim": 128,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "first_k_dense": 1,
    "partial_rotary_factor": 0.5,
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "mrope_section": (8, 12, 12),
    "tie_embeddings": False,
    "vision_depth": 24,
    "vision_embed_dim": 1536,
    "vision_num_heads": 12,
    "vision_intermediate_size": 13696,
    "vision_out_hidden_size": 4096,
    "image_size": 336,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "vision_norm_eps": 1e-5,
    "image_token_id": 151363,
    "video_token_id": 151364,
    "image_start_token_id": 151339,
    "image_end_token_id": 151340,
    "video_start_token_id": 151341,
    "video_end_token_id": 151342,
}

GLM4V_MOE_CONFIG = {
    "glm-4.5v": dict(GLM4V_MOE_COMMON),
}

GLM4V_MOE_WEIGHTS_URLS = {
    "glm-4.5v": {
        "hf_id": "zai-org/GLM-4.5V",
        "gated": False,
        "safetensors": True,
    },
}
