"""Pure Keras 3 layers for the Qwen2 dense LLM (self-contained).

RMSNorm, 1D rotary helpers, grouped-query causal self-attention (q/k/v bias,
bias-free o_proj) with an optional KV cache, the SwiGLU MLP, and the decoder
block. ``keras.ops`` only — runs on TF / Torch / JAX.
"""

import keras
import numpy as np
from keras import layers, ops


def rotate_half(x):
    """Rotate the last dim by halves: ``[-x2, x1]`` (Llama/RoPE convention)."""
    half = x.shape[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary(t, cos, sin):
    """Standard rotary application ``t * cos + rotate_half(t) * sin``."""
    return (t * cos) + (rotate_half(t) * sin)


def rope_cos_sin(position_ids, head_dim, theta):
    """1D rotary tables from position ids.

    Args:
        position_ids: ``(batch, seq)`` int positions.
        head_dim: attention head dim.
        theta: rotary base.

    Returns:
        ``(cos, sin)`` numpy arrays, each ``(batch, seq, head_dim)``.
    """
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = np.asarray(position_ids, dtype="float32")[..., None] * inv_freq
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype("float32"), np.sin(emb).astype("float32")


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2RMSNorm(layers.Layer):
    """RMSNorm: normalize by RMS in float32, then scale."""

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(input_shape[-1],), initializer="ones", trainable=True
        )
        self.built = True

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(variance + self.eps)
        return self.weight * ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2MLP(layers.Layer):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` (bias-free)."""

    def __init__(self, hidden_size, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = layers.Dense(
            intermediate_size, use_bias=False, name="gate_proj"
        )
        self.up_proj = layers.Dense(intermediate_size, use_bias=False, name="up_proj")
        self.down_proj = layers.Dense(hidden_size, use_bias=False, name="down_proj")

    def call(self, x):
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2Attention(layers.Layer):
    """Grouped-query causal self-attention with 1D rotary positions.

    ``q_proj`` / ``k_proj`` / ``v_proj`` carry a bias (Qwen2); ``o_proj`` does
    not. K/V heads (``num_key_value_heads``) are repeated to match Q (GQA). A
    KV cache may be threaded through ``past_key_value`` for incremental decode.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = layers.Dense(
            num_attention_heads * self.head_dim, use_bias=True, name="q_proj"
        )
        self.k_proj = layers.Dense(
            num_key_value_heads * self.head_dim, use_bias=True, name="k_proj"
        )
        self.v_proj = layers.Dense(
            num_key_value_heads * self.head_dim, use_bias=True, name="v_proj"
        )
        self.o_proj = layers.Dense(hidden_size, use_bias=False, name="o_proj")

    def _split_heads(self, x, num_heads):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        return ops.transpose(
            ops.reshape(x, (b, s, num_heads, self.head_dim)), (0, 2, 1, 3)
        )

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]
        query = self._split_heads(self.q_proj(hidden_states), self.num_attention_heads)
        key = self._split_heads(self.k_proj(hidden_states), self.num_key_value_heads)
        value = self._split_heads(self.v_proj(hidden_states), self.num_key_value_heads)

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            key = ops.concatenate([past_k, key], axis=2)
            value = ops.concatenate([past_v, value], axis=2)
        new_key_value = (key, value) if use_cache else None

        if self.num_key_value_groups > 1:
            key = ops.repeat(key, self.num_key_value_groups, axis=1)
            value = ops.repeat(value, self.num_key_value_groups, axis=1)

        attn = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, value)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, q_len, self.num_attention_heads * self.head_dim))
        out = self.o_proj(out)
        return (out, new_key_value) if use_cache else out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2DecoderLayer(layers.Layer):
    """One Qwen2 decoder block: pre-norm GQA attention then pre-norm SwiGLU."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim=None,
        rms_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.input_layernorm = Qwen2RMSNorm(eps=rms_norm_eps, name="input_layernorm")
        self.self_attn = Qwen2Attention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim=self.head_dim,
            name="self_attn",
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            eps=rms_norm_eps, name="post_attention_layernorm"
        )
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, name="mlp")

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states,
            cos,
            sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        new_key_value = None
        if use_cache:
            attn_out, new_key_value = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_key_value) if use_cache else hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "rms_norm_eps": self.rms_norm_eps,
            }
        )
        return config
