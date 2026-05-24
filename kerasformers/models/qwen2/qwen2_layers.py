import keras
from keras import layers, ops


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
class Qwen2Attention(layers.Layer):
    """Grouped-query causal self-attention with 1D rotary positions.

    ``query`` / ``key`` / ``value`` carry a bias (Qwen2); ``output_proj`` does
    not. K/V heads (``num_kv_heads``) are repeated to match the query heads
    (GQA). A KV cache may be threaded through ``past_key_value`` for incremental
    decode.
    """

    def __init__(self, embed_dim, num_heads, num_kv_heads, head_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.query = layers.Dense(
            num_heads * self.head_dim, use_bias=True, name="query"
        )
        self.key = layers.Dense(num_kv_heads * self.head_dim, use_bias=True, name="key")
        self.value = layers.Dense(
            num_kv_heads * self.head_dim, use_bias=True, name="value"
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

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
        query = self._split_heads(self.query(hidden_states), self.num_heads)
        key = self._split_heads(self.key(hidden_states), self.num_kv_heads)
        value = self._split_heads(self.value(hidden_states), self.num_kv_heads)

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        half = self.head_dim // 2
        query = query * cos + (
            ops.concatenate([-query[..., half:], query[..., :half]], axis=-1) * sin
        )
        key = key * cos + (
            ops.concatenate([-key[..., half:], key[..., :half]], axis=-1) * sin
        )

        if past_key_value is not None:
            past_k, past_v = past_key_value
            key = ops.concatenate([past_k, key], axis=2)
            value = ops.concatenate([past_v, value], axis=2)
        new_key_value = (key, value) if use_cache else None

        if self.num_kv_groups > 1:
            key = ops.repeat(key, self.num_kv_groups, axis=1)
            value = ops.repeat(value, self.num_kv_groups, axis=1)

        attn = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, value)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, q_len, self.num_heads * self.head_dim))
        out = self.output_proj(out)
        return (out, new_key_value) if use_cache else out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2DecoderLayer(layers.Layer):
    """One Qwen2 decoder block: pre-norm GQA attention then pre-norm SwiGLU."""

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim=None,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.norm_eps = norm_eps
        self.attention_norm = Qwen2RMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Qwen2Attention(
            embed_dim, num_heads, num_kv_heads, head_dim=self.head_dim, name="attention"
        )
        self.mlp_norm = Qwen2RMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = Qwen2MLP(embed_dim, mlp_dim, name="mlp")

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
        attn_out = self.attention(
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
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_key_value) if use_cache else hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config
