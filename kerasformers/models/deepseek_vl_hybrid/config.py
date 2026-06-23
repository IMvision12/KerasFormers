# DeepSeek-VL Hybrid (7B), model_type "deepseek_vl_hybrid": a SigLIP-L/16 @384
# low-res tower PLUS a SAM/ViTDet-B @1024 high-res tower, fused by a learned
# `alpha` + a 3-way aligner, on a LLaMA-7B decoder. 7B chat and base share the
# architecture (only the trained weights differ). The plain "deepseek_vl" 1.3B
# repos (no high-res branch) live in `deepseek_vl/`; this reuses that folder's
# SigLIP, LLaMA, and tokenizer components.
DEEPSEEK_VL_HYBRID_CONFIG = {
    "deepseek_vl_7b_chat": {
        "vocab_size": 102400,
        "embed_dim": 4096,
        "mlp_dim": 11008,
        "num_layers": 30,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_embeddings": False,
        "vision_embed_dim": 1024,
        "vision_mlp_dim": 4096,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "image_size": 384,
        "patch_size": 16,
        "vision_norm_eps": 1e-6,
        "high_res_embed_dim": 768,
        "high_res_mlp_dim": 3072,
        "high_res_num_layers": 12,
        "high_res_num_heads": 12,
        "high_res_image_size": 1024,
        "high_res_patch_size": 16,
        "high_res_output_channels": 256,
        "high_res_window_size": 14,
        "high_res_global_attn_indexes": (2, 5, 8, 11),
        "high_res_norm_eps": 1e-6,
        "image_token_id": 100015,
    },
    "deepseek_vl_7b_base": {
        "vocab_size": 102400,
        "embed_dim": 4096,
        "mlp_dim": 11008,
        "num_layers": 30,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_embeddings": False,
        "vision_embed_dim": 1024,
        "vision_mlp_dim": 4096,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "image_size": 384,
        "patch_size": 16,
        "vision_norm_eps": 1e-6,
        "high_res_embed_dim": 768,
        "high_res_mlp_dim": 3072,
        "high_res_num_layers": 12,
        "high_res_num_heads": 12,
        "high_res_image_size": 1024,
        "high_res_patch_size": 16,
        "high_res_output_channels": 256,
        "high_res_window_size": 14,
        "high_res_global_attn_indexes": (2, 5, 8, 11),
        "high_res_norm_eps": 1e-6,
        "image_token_id": 100015,
    },
}

DEEPSEEK_VL_HYBRID_WEIGHTS_URLS = {
    "deepseek_vl_7b_chat": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/deepseek_vl_hybrid/deepseek_vl_7b_chat.weights.json"
    },
    "deepseek_vl_7b_base": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/deepseek_vl_hybrid/deepseek_vl_7b_base.weights.json"
    },
}

DEEPSEEK_VL_HYBRID_TOKENIZER_URLS = {
    "deepseek_vl_7b_chat": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/deepseek_vl_hybrid/deepseek_vl_7b_chat_tokenizer.json"
    },
    "deepseek_vl_7b_base": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/deepseek_vl_hybrid/deepseek_vl_7b_base_tokenizer.json"
    },
}
