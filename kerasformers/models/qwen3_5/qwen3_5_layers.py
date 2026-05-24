"""Pure Keras 3 layers for the Qwen3.5 (Qwen3-Next) hybrid LLM — self-contained.

Two token mixers, chosen per layer by ``layer_types``:

* **Gated-DeltaNet linear attention** — a depthwise causal Conv1d (+ SiLU) over
  fused q/k/v, then the gated delta-rule recurrence (decay ``g``, gate ``beta``,
  L2-normed q/k), a gated RMSNorm, and an output projection.
* **Gated full attention** — GQA with per-head QK-norm, *partial* rotary, and a
  sigmoid output gate.

Norms are zero-centered RMSNorm (``(1 + weight)``); the gated norm inside the
DeltaNet uses ``weight`` directly. Everything is ``keras.ops`` (TF/Torch/JAX).
"""

import keras
import numpy as np
from keras import layers, ops


def rotate_half(x):
    half = x.shape[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def rope_cos_sin(position_ids, rotary_dim, theta):
    """1D partial-rotary tables ``(batch, seq, rotary_dim)``."""
    inv_freq = 1.0 / (
        theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
    )
    freqs = np.asarray(position_ids, dtype="float32")[..., None] * inv_freq
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype("float32"), np.sin(emb).astype("float32")


def apply_partial_rotary(t, cos, sin, rotary_dim):
    """Rotate only the first ``rotary_dim`` channels; pass the rest through."""
    t_rot, t_pass = t[..., :rotary_dim], t[..., rotary_dim:]
    rotated = (t_rot * cos) + (rotate_half(t_rot) * sin)
    return ops.concatenate([rotated, t_pass], axis=-1)


def l2norm(x, eps=1e-6):
    return x * ops.rsqrt(ops.sum(ops.square(x), axis=-1, keepdims=True) + eps)


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3_5RMSNorm(layers.Layer):
    """Zero-centered RMSNorm: ``(1 + weight) * x / rms(x)`` (weight init 0)."""

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(input_shape[-1],), initializer="zeros", trainable=True
        )
        self.built = True

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        x = x * ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps)
        return (1.0 + self.weight) * ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3_5RMSNormGated(layers.Layer):
    """Gated RMSNorm: ``weight * (x / rms(x)) * silu(gate)`` (weight init 1)."""

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(input_shape[-1],), initializer="ones", trainable=True
        )
        self.built = True

    def call(self, x, gate):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        x = x * ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps)
        x = self.weight * ops.cast(x, dtype)
        return x * ops.silu(ops.cast(gate, "float32"))

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3_5MLP(layers.Layer):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` (bias-free)."""

    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.gate = layers.Dense(mlp_dim, use_bias=False, name="gate")
        self.up = layers.Dense(mlp_dim, use_bias=False, name="up")
        self.down = layers.Dense(embed_dim, use_bias=False, name="down")

    def call(self, x):
        return self.down(ops.silu(self.gate(x)) * self.up(x))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3_5Attention(layers.Layer):
    """Gated GQA full attention: QK-norm, partial rotary, sigmoid output gate.

    ``query`` emits both the query and a gate (``num_heads * head_dim * 2``);
    the attention output is multiplied by ``sigmoid(gate)`` before ``output_proj``.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.norm_eps = norm_eps
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(
            num_heads * head_dim * 2, use_bias=False, name="query"
        )
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        self.query_norm = Qwen3_5RMSNorm(eps=norm_eps, name="query_norm")
        self.key_norm = Qwen3_5RMSNorm(eps=norm_eps, name="key_norm")

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
        qg = ops.reshape(
            self.query(hidden_states),
            (b, q_len, self.num_heads, self.head_dim * 2),
        )
        query, gate = qg[..., : self.head_dim], qg[..., self.head_dim :]
        gate = ops.reshape(gate, (b, q_len, self.num_heads * self.head_dim))
        query = self.query_norm(query)
        key = self.key_norm(
            ops.reshape(
                self.key(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
            )
        )
        value = ops.reshape(
            self.value(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )

        query = ops.transpose(query, (0, 2, 1, 3))
        key = ops.transpose(key, (0, 2, 1, 3))
        value = ops.transpose(value, (0, 2, 1, 3))

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        query = apply_partial_rotary(query, cos, sin, self.rotary_dim)
        key = apply_partial_rotary(key, cos, sin, self.rotary_dim)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            key = ops.concatenate([past_k, key], axis=2)
            value = ops.concatenate([past_v, value], axis=2)
        new_kv = (key, value) if use_cache else None

        if self.num_kv_groups > 1:
            key = ops.repeat(key, self.num_kv_groups, axis=1)
            value = ops.repeat(value, self.num_kv_groups, axis=1)

        attn = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, value)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)),
            (b, q_len, self.num_heads * self.head_dim),
        )
        out = out * ops.sigmoid(gate)
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "rotary_dim": self.rotary_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3_5GatedDeltaNet(layers.Layer):
    """Gated-DeltaNet linear attention (conv1d + delta-rule recurrence)."""

    def __init__(
        self,
        embed_dim,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_dim=4,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_dim = conv_kernel_dim
        self.norm_eps = norm_eps
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.kv_ratio = num_v_heads // num_k_heads

        self.in_proj_qkv = layers.Dense(
            self.conv_dim, use_bias=False, name="in_proj_qkv"
        )
        self.in_proj_z = layers.Dense(self.value_dim, use_bias=False, name="in_proj_z")
        self.in_proj_b = layers.Dense(num_v_heads, use_bias=False, name="in_proj_b")
        self.in_proj_a = layers.Dense(num_v_heads, use_bias=False, name="in_proj_a")
        self.norm = Qwen3_5RMSNormGated(eps=norm_eps, name="norm")
        self.out_proj = layers.Dense(embed_dim, use_bias=False, name="out_proj")

    def build(self, input_shape):
        self.conv_weight = self.add_weight(
            name="conv_weight",
            shape=(self.conv_dim, self.conv_kernel_dim),
            initializer="zeros",
            trainable=True,
        )
        self.dt_bias = self.add_weight(
            name="dt_bias",
            shape=(self.num_v_heads,),
            initializer="zeros",
            trainable=True,
        )
        self.A_log = self.add_weight(
            name="A_log", shape=(self.num_v_heads,), initializer="zeros", trainable=True
        )
        self.built = True

    def _causal_conv(self, x, conv_state=None):
        k = self.conv_kernel_dim
        if conv_state is not None:
            x_pad = ops.concatenate([conv_state, x], axis=1)
        else:
            x_pad = ops.concatenate(
                [ops.zeros((ops.shape(x)[0], k - 1, self.conv_dim), dtype=x.dtype), x],
                axis=1,
            )
        seq = ops.shape(x)[1]
        out = 0.0
        for j in range(k):
            out = out + x_pad[:, j : j + seq, :] * self.conv_weight[:, j]
        new_state = x_pad[:, -(k - 1) :, :] if (k - 1) > 0 else None
        return ops.silu(out), new_state

    def _delta_rule(self, q, k, v, g, beta, init_state=None):
        q = l2norm(q) * (self.head_k_dim**-0.5)
        k = l2norm(k)
        b = ops.shape(q)[0]
        seq = q.shape[1]
        state = (
            init_state
            if init_state is not None
            else ops.zeros(
                (b, self.num_v_heads, self.head_k_dim, self.head_v_dim), dtype=q.dtype
            )
        )
        outs = []
        for t in range(seq):
            q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
            g_t = ops.exp(g[:, t])[:, :, None, None]
            beta_t = beta[:, t][:, :, None]
            state = state * g_t
            kv_mem = ops.sum(state * k_t[..., None], axis=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t[..., None] * delta[:, :, None, :]
            outs.append(ops.sum(state * q_t[..., None], axis=-2))
        core = ops.stack(outs, axis=1)
        return core, state

    def call(self, hidden_states, past_key_value=None, use_cache=False):
        b = ops.shape(hidden_states)[0]
        seq = ops.shape(hidden_states)[1]
        conv_state = past_key_value[0] if past_key_value is not None else None
        rec_state = past_key_value[1] if past_key_value is not None else None

        mixed = self.in_proj_qkv(hidden_states)
        mixed, new_conv_state = self._causal_conv(mixed, conv_state)
        query = ops.reshape(
            mixed[..., : self.key_dim], (b, seq, self.num_k_heads, self.head_k_dim)
        )
        key = ops.reshape(
            mixed[..., self.key_dim : self.key_dim * 2],
            (b, seq, self.num_k_heads, self.head_k_dim),
        )
        value = ops.reshape(
            mixed[..., self.key_dim * 2 :], (b, seq, self.num_v_heads, self.head_v_dim)
        )

        z = ops.reshape(
            self.in_proj_z(hidden_states), (b, seq, self.num_v_heads, self.head_v_dim)
        )
        beta = ops.sigmoid(self.in_proj_b(hidden_states))
        a = self.in_proj_a(hidden_states)
        g = -ops.exp(ops.cast(self.A_log, "float32")) * ops.softplus(
            ops.cast(a, "float32") + ops.cast(self.dt_bias, "float32")
        )

        if self.kv_ratio > 1:
            query = ops.repeat(query, self.kv_ratio, axis=2)
            key = ops.repeat(key, self.kv_ratio, axis=2)

        core, new_rec_state = self._delta_rule(query, key, value, g, beta, rec_state)
        core = self.norm(core, z)
        core = ops.reshape(core, (b, seq, self.value_dim))
        out = self.out_proj(core)
        return (out, (new_conv_state, new_rec_state)) if use_cache else out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_k_heads": self.num_k_heads,
                "num_v_heads": self.num_v_heads,
                "head_k_dim": self.head_k_dim,
                "head_v_dim": self.head_v_dim,
                "conv_kernel_dim": self.conv_kernel_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3_5DecoderLayer(layers.Layer):
    """Hybrid decoder block — linear-attention or full-attention token mixer."""

    def __init__(self, config, layer_type, **kwargs):
        super().__init__(**kwargs)
        self.config_dict = dict(config)
        self.layer_type = layer_type
        c = config
        eps = c["norm_eps"]
        self.attention_norm = Qwen3_5RMSNorm(eps=eps, name="attention_norm")
        self.mlp_norm = Qwen3_5RMSNorm(eps=eps, name="mlp_norm")
        self.mlp = Qwen3_5MLP(c["embed_dim"], c["mlp_dim"], name="mlp")
        if layer_type == "full_attention":
            self.attention = Qwen3_5Attention(
                c["embed_dim"],
                c["num_heads"],
                c["num_kv_heads"],
                c["head_dim"],
                c["rotary_dim"],
                eps,
                name="attention",
            )
        else:
            self.linear_attn = Qwen3_5GatedDeltaNet(
                c["embed_dim"],
                c["linear_num_key_heads"],
                c["linear_num_value_heads"],
                c["linear_key_head_dim"],
                c["linear_value_head_dim"],
                c["linear_conv_kernel_dim"],
                eps,
                name="linear_attn",
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
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        new_state = None
        if self.layer_type == "full_attention":
            out = self.attention(
                hidden_states,
                cos,
                sin,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
        else:
            out = self.linear_attn(
                hidden_states, past_key_value=past_key_value, use_cache=use_cache
            )
        if use_cache:
            out, new_state = out
        hidden_states = residual + out
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_state) if use_cache else hidden_states

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config_dict, "layer_type": self.layer_type})
        return config
