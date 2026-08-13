from kerasformers.base import BaseConfig


class Qwen3Config(BaseConfig):
    r"""Configuration for Qwen3 (dense): [`Qwen3Model`] and [`Qwen3Generate`].

    A Qwen3 decoder: grouped-query attention with per-head QK-norm and bias-free
    QKV projections, SwiGLU MLP, RMSNorm, and 1D rotary positions.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 1024):
            Model / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 3072):
            SwiGLU hidden width per layer.
        num_layers (`int`, *optional*, defaults to 28):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Query heads per layer.
        num_kv_heads (`int`, *optional*, defaults to 8):
            Key/value heads per layer (GQA).
        head_dim (`int`, *optional*, defaults to 128):
            Per-head dim.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon (shared by the per-head QK-norms).
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Rotary base frequency.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the LM head is tied to the token embedding."""

    model_type = "qwen3"

    vocab_size: int = 151936
    embed_dim: int = 1024
    mlp_dim: int = 3072
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_embeddings: bool = True


QWEN3_CONFIG = {
    "qwen3-0.6b": {
        "vocab_size": 151936,
        "embed_dim": 1024,
        "mlp_dim": 3072,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_embeddings": True,
    },
    "qwen3-0.6b-base": {
        "vocab_size": 151936,
        "embed_dim": 1024,
        "mlp_dim": 3072,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_embeddings": True,
    },
    "qwen3-1.7b": {
        "vocab_size": 151936,
        "embed_dim": 2048,
        "mlp_dim": 6144,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_embeddings": True,
    },
    "qwen3-1.7b-base": {
        "vocab_size": 151936,
        "embed_dim": 2048,
        "mlp_dim": 6144,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_embeddings": True,
    },
    "qwen3-4b": {
        "vocab_size": 151936,
        "embed_dim": 2560,
        "mlp_dim": 9728,
        "num_layers": 36,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_embeddings": True,
    },
    "qwen3-4b-base": {
        "vocab_size": 151936,
        "embed_dim": 2560,
        "mlp_dim": 9728,
        "num_layers": 36,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_embeddings": True,
    },
    "qwen3-8b": {
        "vocab_size": 151936,
        "embed_dim": 4096,
        "mlp_dim": 12288,
        "num_layers": 36,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_embeddings": False,
    },
    "qwen3-8b-base": {
        "vocab_size": 151936,
        "embed_dim": 4096,
        "mlp_dim": 12288,
        "num_layers": 36,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_embeddings": False,
    },
    "qwen3-14b": {
        "vocab_size": 151936,
        "embed_dim": 5120,
        "mlp_dim": 17408,
        "num_layers": 40,
        "num_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_embeddings": False,
    },
    "qwen3-14b-base": {
        "vocab_size": 151936,
        "embed_dim": 5120,
        "mlp_dim": 17408,
        "num_layers": 40,
        "num_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_embeddings": False,
    },
    "qwen3-32b": {
        "vocab_size": 151936,
        "embed_dim": 5120,
        "mlp_dim": 25600,
        "num_layers": 64,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_embeddings": False,
    },
}

QWEN3_WEIGHTS_URLS = {
    "qwen3-0.6b": {"hf_id": "Qwen/Qwen3-0.6B", "gated": False, "safetensors": True},
    "qwen3-0.6b-base": {
        "hf_id": "Qwen/Qwen3-0.6B-Base",
        "gated": False,
        "safetensors": True,
    },
    "qwen3-1.7b": {"hf_id": "Qwen/Qwen3-1.7B", "gated": False, "safetensors": True},
    "qwen3-1.7b-base": {
        "hf_id": "Qwen/Qwen3-1.7B-Base",
        "gated": False,
        "safetensors": True,
    },
    "qwen3-4b": {"hf_id": "Qwen/Qwen3-4B", "gated": False, "safetensors": True},
    "qwen3-4b-base": {
        "hf_id": "Qwen/Qwen3-4B-Base",
        "gated": False,
        "safetensors": True,
    },
    "qwen3-8b": {"hf_id": "Qwen/Qwen3-8B", "gated": False, "safetensors": True},
    "qwen3-8b-base": {
        "hf_id": "Qwen/Qwen3-8B-Base",
        "gated": False,
        "safetensors": True,
    },
    "qwen3-14b": {"hf_id": "Qwen/Qwen3-14B", "gated": False, "safetensors": True},
    "qwen3-14b-base": {
        "hf_id": "Qwen/Qwen3-14B-Base",
        "gated": False,
        "safetensors": True,
    },
    "qwen3-32b": {"hf_id": "Qwen/Qwen3-32B", "gated": False, "safetensors": True},
}
