# Gemma 4 "unified" family (google/gemma-4-12B). Unlike the "gemma4" family
# (NaViT vision tower + USM audio conformer), the unified checkpoints are
# encoder-free: images become raw 48px merged pixel patches projected by a
# Dense + factorized 2D position embedding, and audio becomes raw 640-sample
# waveform frames projected straight through an RMSNorm + Dense. The text tower
# is the plain Gemma 4 dense decoder (no Per-Layer Embeddings, no MoE) with
# global K=V attention and a learned per-layer scalar, so it reuses
# :class:`~kerasformers.models.gemma4.gemma4_model.Gemma4Model`.

# Text configs forwarded to Gemma4Model. The 12B has no KV sharing
# (num_kv_shared_layers 0) but keeps global K=V attention (k_eq_v).
GEMMA4_UNIFIED_CONFIG = {
    "gemma-4-12b": {
        "embed_dim": 3840,
        "mlp_dim": 15360,
        "num_layers": 48,
        "num_heads": 16,
        "num_kv_heads": 8,
        "num_global_kv_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "k_eq_v": True,
        "enable_moe": False,
    },
    "gemma-4-12b-it": {
        "embed_dim": 3840,
        "mlp_dim": 15360,
        "num_layers": 48,
        "num_heads": 16,
        "num_kv_heads": 8,
        "num_global_kv_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "k_eq_v": True,
        "enable_moe": False,
    },
}

GEMMA4_UNIFIED_WEIGHTS_URLS = {
    "gemma-4-12b": {"hf_id": "google/gemma-4-12B", "gated": False, "safetensors": True},
    "gemma-4-12b-it": {
        "hf_id": "google/gemma-4-12B-it",
        "gated": False,
        "safetensors": True,
    },
}

# Encoder-free vision embedder. ``model_patch_size = patch_size *
# pooling_kernel_size`` (48), so each merged patch carries ``model_patch_size**2
# * 3`` (6912) raw pixel channels. ``mm_posemb_size`` is the length of the
# factorized 2D position table (shape ``(mm_posemb_size, 2, mm_embed_dim)``).
GEMMA4_UNIFIED_VISION_CONFIG = {
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "mm_embed_dim": 3840,
    "mm_posemb_size": 1120,
    "output_proj_dims": 3840,
    "eps": 1e-6,
}

# Encoder-free audio embedder. Each soft token is a raw 40ms waveform frame of
# ``audio_embed_dim`` (640) samples at 16 kHz, projected straight to text space.
GEMMA4_UNIFIED_AUDIO_CONFIG = {
    "audio_embed_dim": 640,
    "output_proj_dims": 640,
    "eps": 1e-6,
}

# The full multimodal generator (text + encoder-free vision + audio), mirroring
# transformers' Gemma4UnifiedForConditionalGeneration.
GEMMA4_UNIFIED_GENERATE_CONFIG = {
    "gemma-4-12b": {
        "text_config": dict(GEMMA4_UNIFIED_CONFIG["gemma-4-12b"]),
        "vision_config": dict(GEMMA4_UNIFIED_VISION_CONFIG),
        "audio_config": dict(GEMMA4_UNIFIED_AUDIO_CONFIG),
        "image_token_id": 258880,
        "video_token_id": 258884,
        "audio_token_id": 258881,
        "use_bidirectional_vision": True,
    },
    "gemma-4-12b-it": {
        "text_config": dict(GEMMA4_UNIFIED_CONFIG["gemma-4-12b-it"]),
        "vision_config": dict(GEMMA4_UNIFIED_VISION_CONFIG),
        "audio_config": dict(GEMMA4_UNIFIED_AUDIO_CONFIG),
        "image_token_id": 258880,
        "video_token_id": 258884,
        "audio_token_id": 258881,
        "use_bidirectional_vision": True,
    },
}

GEMMA4_UNIFIED_GENERATE_WEIGHTS_URLS = dict(GEMMA4_UNIFIED_WEIGHTS_URLS)
