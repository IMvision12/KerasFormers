LOCATEANYTHING_CONFIG = {
    "locateanything-3b": {
        # --- text backbone (Qwen2.5-3B-Instruct) ---
        "vocab_size": 152681,
        "embed_dim": 2048,
        "mlp_dim": 11008,
        "num_layers": 36,
        "num_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "tie_embeddings": True,
        # --- vision encoder (MoonViT-SO-400M) ---
        "vision_embed_dim": 1152,
        "vision_depth": 27,
        "vision_num_heads": 16,
        "vision_mlp_dim": 4304,
        "vision_patch_size": 14,
        "vision_init_pos_h": 64,
        "vision_init_pos_w": 64,
        "merge_kernel": (2, 2),
        "vision_rope_theta": 10000.0,
        # --- multimodal / special tokens ---
        "image_token_index": 151665,
        "box_start_token_id": 151668,
        "box_end_token_id": 151669,
        "ref_start_token_id": 151672,
        "ref_end_token_id": 151673,
        "coord_start_token_id": 151677,
        "coord_end_token_id": 152677,
        "none_token_id": 4064,
        "text_mask_token_id": 151676,
        "mlp_connector_layers": 2,
        # --- parallel box decoding (MTP) ---
        "block_size": 6,
    },
}

LOCATEANYTHING_WEIGHTS_URLS = {
    "locateanything-3b": {
        "hf_id": "nvidia/LocateAnything-3B",
        "gated": False,
        "safetensors": True,
    },
}
