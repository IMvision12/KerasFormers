COHERE2_VISION_CONFIG = {
    "command-a-vision-07-2025": {
        "vocab_size": 256000,
        "embed_dim": 12288,
        "num_layers": 64,
        "num_heads": 96,
        "num_kv_heads": 8,
        "head_dim": 128,
        "mlp_dim": 36864,
        "sliding_window": 4096,
        "sliding_window_pattern": 4,
        "norm_eps": 1e-5,
        "rope_theta": 50000.0,
        "logit_scale": 0.25,
        "tie_embeddings": True,
        "vision_embed_dim": 1152,
        "vision_mlp_dim": 4304,
        "vision_num_layers": 27,
        "vision_num_heads": 16,
        "image_size": 512,
        "patch_size": 16,
        "vision_norm_eps": 1e-6,
        "downsample_factor": 2,
        "alignment_intermediate_size": 36864,
        "image_token_id": 255036,
    },
}

COHERE2_VISION_WEIGHTS_URLS = {
    "command-a-vision-07-2025": {
        "hf_id": "CohereLabs/command-a-vision-07-2025",
        "gated": True,
        "safetensors": True,
    },
}
