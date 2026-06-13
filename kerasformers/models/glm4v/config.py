GLM4V_TOKENS = {
    "image_token_id": 151343,
    "video_token_id": 151344,
    "image_start_token_id": 151339,
    "image_end_token_id": 151340,
    "video_start_token_id": 151341,
    "video_end_token_id": 151342,
}

GLM4V_BASE = {
    "vocab_size": 151552,
    "embed_dim": 4096,
    "mlp_dim": 13696,
    "num_layers": 40,
    "num_heads": 32,
    "num_kv_heads": 2,
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
    **GLM4V_TOKENS,
}

GLM4V_CONFIG = {
    "glm-4.1v-9b-thinking": dict(GLM4V_BASE),
    "glm-4.1v-9b-base": dict(GLM4V_BASE),
}

GLM4V_WEIGHTS_URLS = {
    "glm-4.1v-9b-thinking": {
        "hf_id": "zai-org/GLM-4.1V-9B-Thinking",
        "gated": False,
        "safetensors": True,
    },
    "glm-4.1v-9b-base": {
        "hf_id": "zai-org/GLM-4.1V-9B-Base",
        "gated": False,
        "safetensors": True,
    },
}
