"""Pure Keras 3 layers for the Qwen3 dense LLM (self-contained).

Like Qwen2 but the attention applies per-head RMSNorm to q and k (pre-RoPE) and
has no q/k/v bias. RMSNorm, 1D rotary, GQA, SwiGLU.
"""

import keras
import numpy as np
from keras import layers, ops


def rotate_half(x):
    """Rotate the last dim by halves: ``[-x2, x1]``."""
    half = x.shape[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary(t, cos, sin):
    return (t * cos) + (rotate_half(t) * sin)


def rope_cos_sin(position_ids, head_dim, theta):
    """1D rotary tables ``(batch, seq, head_dim)`` from position ids."""
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = np.asarray(position_ids, dtype="float32")[..., None] * inv_freq
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype("float32"), np.sin(emb).astype("float32")


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3RMSNorm(layers.Layer):
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
class Qwen3MLP(layers.Layer):
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
class Qwen3Attention(layers.Layer):
    """GQA causal self-attention with per-head QK-norm, no qkv bias, 1D rotary."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim**-0.5
        self.q_proj = layers.Dense(
            num_attention_heads * head_dim, use_bias=False, name="q_proj"
        )
        self.k_proj = layers.Dense(
            num_key_value_heads * head_dim, use_bias=False, name="k_proj"
        )
        self.v_proj = layers.Dense(
            num_key_value_heads * head_dim, use_bias=False, name="v_proj"
        )
        self.o_proj = layers.Dense(hidden_size, use_bias=False, name="o_proj")
        self.q_norm = Qwen3RMSNorm(eps=rms_norm_eps, name="q_norm")
        self.k_norm = Qwen3RMSNorm(eps=rms_norm_eps, name="k_norm")

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
        q = self.q_norm(
            ops.reshape(
                self.q_proj(hidden_states),
                (b, q_len, self.num_attention_heads, self.head_dim),
            )
        )
        k = self.k_norm(
            ops.reshape(
                self.k_proj(hidden_states),
                (b, q_len, self.num_key_value_heads, self.head_dim),
            )
        )
        v = ops.reshape(
            self.v_proj(hidden_states),
            (b, q_len, self.num_key_value_heads, self.head_dim),
        )
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = ops.concatenate([past_k, k], axis=2)
            v = ops.concatenate([past_v, v], axis=2)
        new_kv = (k, v) if use_cache else None

        if self.num_key_value_groups > 1:
            k = ops.repeat(k, self.num_key_value_groups, axis=1)
            v = ops.repeat(v, self.num_key_value_groups, axis=1)

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, v)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)),
            (b, q_len, self.num_attention_heads * self.head_dim),
        )
        out = self.o_proj(out)
        return (out, new_kv) if use_cache else out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "rms_norm_eps": self.rms_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3DecoderLayer(layers.Layer):
    """One Qwen3 decoder block: pre-norm QK-norm attention then pre-norm SwiGLU."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.input_layernorm = Qwen3RMSNorm(eps=rms_norm_eps, name="input_layernorm")
        self.self_attn = Qwen3Attention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            name="self_attn",
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            eps=rms_norm_eps, name="post_attention_layernorm"
        )
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, name="mlp")

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
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_kv) if use_cache else hidden_states

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
