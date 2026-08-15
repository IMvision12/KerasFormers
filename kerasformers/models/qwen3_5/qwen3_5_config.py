from kerasformers.base import BaseConfig

QWEN3_5_CONFIG = {
    "qwen3.5-0.8b": {
        "vocab_size": 248320,
        "embed_dim": 1024,
        "mlp_dim": 3584,
        "num_layers": 24,
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 256,
        "norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_embeddings": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
    },
    "qwen3.5-0.8b-base": {
        "vocab_size": 248320,
        "embed_dim": 1024,
        "mlp_dim": 3584,
        "num_layers": 24,
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 256,
        "norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_embeddings": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
    },
    "qwen3.5-2b": {
        "vocab_size": 248320,
        "embed_dim": 2048,
        "mlp_dim": 6144,
        "num_layers": 24,
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 256,
        "norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_embeddings": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
    },
    "qwen3.5-2b-base": {
        "vocab_size": 248320,
        "embed_dim": 2048,
        "mlp_dim": 6144,
        "num_layers": 24,
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 256,
        "norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_embeddings": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
    },
    "qwen3.5-4b": {
        "vocab_size": 248320,
        "embed_dim": 2560,
        "mlp_dim": 9216,
        "num_layers": 32,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 256,
        "norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_embeddings": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
    },
    "qwen3.5-4b-base": {
        "vocab_size": 248320,
        "embed_dim": 2560,
        "mlp_dim": 9216,
        "num_layers": 32,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 256,
        "norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_embeddings": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
    },
    "qwen3.5-9b": {
        "vocab_size": 248320,
        "embed_dim": 4096,
        "mlp_dim": 12288,
        "num_layers": 32,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 256,
        "norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_embeddings": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
    },
    "qwen3.5-9b-base": {
        "vocab_size": 248320,
        "embed_dim": 4096,
        "mlp_dim": 12288,
        "num_layers": 32,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 256,
        "norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_embeddings": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
    },
    "qwen3.5-27b": {
        "vocab_size": 248320,
        "embed_dim": 5120,
        "mlp_dim": 17408,
        "num_layers": 64,
        "num_heads": 24,
        "num_kv_heads": 4,
        "head_dim": 256,
        "norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_embeddings": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
    },
}

QWEN3_5_WEIGHTS_URLS = {
    "qwen3.5-0.8b": {"hf_id": "Qwen/Qwen3.5-0.8B", "gated": False, "safetensors": True},
    "qwen3.5-0.8b-base": {
        "hf_id": "Qwen/Qwen3.5-0.8B-Base",
        "gated": False,
        "safetensors": True,
    },
    "qwen3.5-2b": {"hf_id": "Qwen/Qwen3.5-2B", "gated": False, "safetensors": True},
    "qwen3.5-2b-base": {
        "hf_id": "Qwen/Qwen3.5-2B-Base",
        "gated": False,
        "safetensors": True,
    },
    "qwen3.5-4b": {"hf_id": "Qwen/Qwen3.5-4B", "gated": False, "safetensors": True},
    "qwen3.5-4b-base": {
        "hf_id": "Qwen/Qwen3.5-4B-Base",
        "gated": False,
        "safetensors": True,
    },
    "qwen3.5-9b": {"hf_id": "Qwen/Qwen3.5-9B", "gated": False, "safetensors": True},
    "qwen3.5-9b-base": {
        "hf_id": "Qwen/Qwen3.5-9B-Base",
        "gated": False,
        "safetensors": True,
    },
    "qwen3.5-27b": {"hf_id": "Qwen/Qwen3.5-27B", "gated": False, "safetensors": True},
}


class Qwen3_5TextConfig(BaseConfig):
    r"""Text-decoder config for the dense Qwen3.5 VLM (the ``text_config`` sub-config).

    The dense Qwen3.5 hybrid decoder: mostly Gated-DeltaNet linear-attention layers with
    a gated full-attention block every ``full_attention_interval`` (GQA, per-head QK-norm,
    partial-rotary interleaved M-RoPE), each with a dense GeGLU MLP. Identical to the
    text-only Qwen3.5 backbone, plus ``mrope_section`` for the multimodal position tables.

    Args mirror the flat ``QWEN3_5_CONFIG`` entries; see [`Qwen3_5TextGenerate`]."""

    model_type = "qwen3_5_text"

    vocab_size: int = 248320
    embed_dim: int = 5120
    mlp_dim: int = 17408
    num_layers: int = 64
    num_heads: int = 24
    num_kv_heads: int = 4
    head_dim: int = 256
    norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple = (11, 11, 10)
    tie_embeddings: bool = False
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48


class Qwen3_5VisionConfig(BaseConfig):
    r"""Vision-tower config for the dense Qwen3.5 VLM (the ``vision_config`` sub-config).

    The Qwen3-VL ViT (no DeepStack): full attention over the packed patch sequence,
    learned (bilinearly interpolated) position embeddings, GELU MLP blocks, and a 2x2
    spatial-merge projector to the text ``out_dim``."""

    model_type = "qwen3_5_vision"

    depth: int = 27
    embed_dim: int = 1152
    mlp_dim: int = 4304
    num_heads: int = 16
    out_dim: int = 5120
    act: str = "gelu_pytorch_tanh"
    num_position_embeddings: int = 2304
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    in_channels: int = 3


class Qwen3_5Config(BaseConfig):
    r"""Configuration for the dense Qwen3.5 VLM: [`Qwen3_5VLModel`] and
    [`Qwen3_5ConditionalGenerate`].

    A composite config: the dense hybrid text decoder lives in a [`Qwen3_5TextConfig`]
    (``text_config``) and the ViT in a [`Qwen3_5VisionConfig`] (``vision_config``); the
    four vision token ids are the top-level image/video glue. Flattened to the model
    constructor with the ``vision_`` prefix on the vision fields, except the geometry
    fields which keep their own name."""

    model_type = "qwen3_5"

    sub_configs = {
        "text_config": Qwen3_5TextConfig,
        "vision_config": Qwen3_5VisionConfig,
    }
    sub_config_prefixes = {"text_config": "", "vision_config": "vision_"}
    group_extras = {
        "vision_config": (
            "num_position_embeddings",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "in_channels",
        )
    }

    text_config: Qwen3_5TextConfig | dict | None = None
    vision_config: Qwen3_5VisionConfig | dict | None = None
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054


QWEN3_5_TOKENS = {
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
}
