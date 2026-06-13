import keras
from keras import layers, ops


def rotate_half_interleaved(x):
    # Cohere interleaved rotate: out = [-x1, x0, -x3, x2, ...] over channel pairs.
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    stacked = ops.stack([-x2, x1], axis=-1)
    return ops.reshape(stacked, ops.shape(x))


def apply_cohere_rope(x, cos, sin):
    # ``cos`` / ``sin`` are repeat-interleaved (one angle per channel pair). The
    # rotation runs in float32 to match the reference, then casts back.
    dtype = x.dtype
    x = ops.cast(x, "float32")
    cos = ops.cast(cos, "float32")
    sin = ops.cast(sin, "float32")
    out = x * cos + rotate_half_interleaved(x) * sin
    return ops.cast(out, dtype)


@keras.saving.register_keras_serializable(package="kerasformers")
class CohereLayerNorm(layers.Layer):
    """Cohere LayerNorm: mean-centered, bias-free, ones-init weight.

    Unlike RMSNorm this subtracts the mean — ``(x - mean) * rsqrt(var + eps)``
    — then scales by a learned weight. The statistics are taken over the last
    axis in float32. Used for the input norm, the final norm, and (with a
    per-head ``(num_heads, head_dim)`` weight) the optional QK-norm.

    Args:
        eps: Variance epsilon. Defaults to ``1e-5``.
    """

    def __init__(self, eps=1e-5, weight_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.weight_shape = None if weight_shape is None else tuple(weight_shape)

    def build(self, input_shape):
        shape = self.weight_shape or (input_shape[-1],)
        self.weight = self.add_weight(
            name="weight", shape=shape, initializer="ones", trainable=True
        )
        self.built = True

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        mean = ops.mean(x, axis=-1, keepdims=True)
        variance = ops.mean(ops.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * ops.rsqrt(variance + self.eps)
        return ops.cast(ops.cast(self.weight, "float32") * x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps, "weight_shape": self.weight_shape})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class CohereMLP(layers.Layer):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))`` (bias-free)."""

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
class CohereAttention(layers.Layer):
    """Cohere grouped-query attention with interleaved rope + optional QK-norm.

    Bias-free (or biased, per ``attention_bias``) ``query`` / ``key`` /
    ``value`` / ``output_proj``. When ``use_qk_norm`` the reshaped per-head
    query and key are LayerNorm'd (per-head ``(num_heads, head_dim)`` weight)
    before rotary. Rotary is Cohere's interleaved variant, applied in float32.

    Args:
        embed_dim: Model width.
        num_heads / num_kv_heads / head_dim: Attention geometry.
        use_qk_norm: Apply the per-head LayerNorm on q/k.
        norm_eps: QK-norm epsilon.
        attention_bias: Whether the projections carry bias.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        use_qk_norm=False,
        norm_eps=1e-5,
        attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm
        self.norm_eps = norm_eps
        self.attention_bias = attention_bias
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(
            num_heads * head_dim, use_bias=attention_bias, name="query"
        )
        self.key = layers.Dense(
            num_kv_heads * head_dim, use_bias=attention_bias, name="key"
        )
        self.value = layers.Dense(
            num_kv_heads * head_dim, use_bias=attention_bias, name="value"
        )
        self.output_proj = layers.Dense(
            embed_dim, use_bias=attention_bias, name="output_proj"
        )
        if use_qk_norm:
            self.query_norm = CohereLayerNorm(
                eps=norm_eps, weight_shape=(num_heads, head_dim), name="query_norm"
            )
            self.key_norm = CohereLayerNorm(
                eps=norm_eps, weight_shape=(num_kv_heads, head_dim), name="key_norm"
            )

    def project(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.query(hidden_states), (b, s, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.key(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.value(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
        )
        if self.use_qk_norm:
            q = self.query_norm(q)
            k = self.key_norm(k)
        return (
            ops.transpose(q, (0, 2, 1, 3)),
            ops.transpose(k, (0, 2, 1, 3)),
            ops.transpose(v, (0, 2, 1, 3)),
        )

    def attend(self, q, k, v, attention_mask):
        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + ops.cast(attention_mask, attn.dtype)
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        return ops.matmul(attn, v)

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q, k, v = self.project(hidden_states)
        cos_e = ops.expand_dims(cos, axis=1)
        sin_e = ops.expand_dims(sin, axis=1)
        q = apply_cohere_rope(q, cos_e, sin_e)
        k = apply_cohere_rope(k, cos_e, sin_e)
        new_kv = (k, v) if use_cache else None
        out = self.attend(q, k, v, attention_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, s, self.num_heads * self.head_dim)
        )
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        b = ops.shape(hidden_states)[0]
        q, k, v = self.project(hidden_states)
        cos_e = ops.expand_dims(cos, axis=1)
        sin_e = ops.expand_dims(sin, axis=1)
        q = apply_cohere_rope(q, cos_e, sin_e)
        k = apply_cohere_rope(k, cos_e, sin_e)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        out = self.attend(q, cache_k, cache_v, key_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, 1, self.num_heads * self.head_dim)
        )
        return self.output_proj(out), cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "use_qk_norm": self.use_qk_norm,
                "norm_eps": self.norm_eps,
                "attention_bias": self.attention_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class CohereDecoderLayer(layers.Layer):
    """One Cohere block: parallel attention + MLP off a single input norm.

    ``h = x + attention(input_norm(x)) + mlp(input_norm(x))`` — the attention
    and MLP read the *same* normed input and both add to the residual (no
    post-attention norm), the PaLM/GPT-J parallel formulation.

    Args:
        embed_dim / mlp_dim / num_heads / num_kv_heads / head_dim: Dims.
        use_qk_norm: Per-head QK LayerNorm.
        norm_eps: LayerNorm epsilon.
        attention_bias: Attention projection bias.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        use_qk_norm=False,
        norm_eps=1e-5,
        attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm
        self.norm_eps = norm_eps
        self.attention_bias = attention_bias
        self.input_layernorm = CohereLayerNorm(eps=norm_eps, name="input_layernorm")
        self.attention = CohereAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            use_qk_norm,
            norm_eps,
            attention_bias,
            name="attention",
        )
        self.mlp = CohereMLP(embed_dim, mlp_dim, name="mlp")

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.attention(
            normed, cos, sin, attention_mask=attention_mask, use_cache=use_cache
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        out = residual + attn_out + self.mlp(normed)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out, cache_k, cache_v = self.attention.decode_step(
            normed, cos, sin, cache_k, cache_v, write_pos, key_mask
        )
        out = residual + attn_out + self.mlp(normed)
        return out, cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "use_qk_norm": self.use_qk_norm,
                "norm_eps": self.norm_eps,
                "attention_bias": self.attention_bias,
            }
        )
        return config
