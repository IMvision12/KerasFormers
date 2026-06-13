import keras
from keras import layers, ops

from kerasformers.models.cohere.cohere_layers import (
    CohereLayerNorm,
    CohereMLP,
    apply_cohere_rope,
)


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2LayerNorm(CohereLayerNorm):
    """Cohere2 LayerNorm (identical to :class:`CohereLayerNorm`)."""


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2MLP(CohereMLP):
    """Cohere2 SwiGLU MLP (identical to :class:`CohereMLP`)."""


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2Attention(layers.Layer):
    """Cohere2 grouped-query attention: NoPE on full layers, rope on sliding.

    No QK-norm. Rotary (Cohere interleaved) is applied **only** on the sliding-
    window layers; the full-attention layers run without positional encoding
    (NoPE). The model passes the matching mask (sliding or full causal).

    Args:
        embed_dim / num_heads / num_kv_heads / head_dim: Geometry.
        use_rope: Apply rotary (True on sliding layers, False on full/NoPE).
        attention_bias: Whether the projections carry bias.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        use_rope,
        attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
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

    def project(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q = ops.transpose(
            ops.reshape(
                self.query(hidden_states), (b, s, self.num_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        k = ops.transpose(
            ops.reshape(
                self.key(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        v = ops.transpose(
            ops.reshape(
                self.value(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        return q, k, v

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
        if self.use_rope:
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
        if self.use_rope:
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
                "use_rope": self.use_rope,
                "attention_bias": self.attention_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2DecoderLayer(layers.Layer):
    """One Cohere2 block: parallel attention + MLP off a single input norm.

    ``h = x + attention(input_norm(x)) + mlp(input_norm(x))``. ``layer_type``
    selects sliding vs full attention (the latter runs NoPE); the model feeds
    the matching mask and only rope-enables the sliding layers.

    Args:
        embed_dim / mlp_dim / num_heads / num_kv_heads / head_dim: Dims.
        layer_type: ``"sliding_attention"`` or ``"full_attention"``.
        norm_eps: LayerNorm epsilon.
        attention_bias: Attention projection bias.
        mlp_cls: Optional MLP/MoE class override (used by Cohere2-MoE).
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        layer_type,
        norm_eps=1e-5,
        attention_bias=False,
        mlp=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_type = layer_type
        self.norm_eps = norm_eps
        self.attention_bias = attention_bias
        self.input_layernorm = Cohere2LayerNorm(eps=norm_eps, name="input_layernorm")
        self.attention = Cohere2Attention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            use_rope=layer_type == "sliding_attention",
            attention_bias=attention_bias,
            name="attention",
        )
        self.mlp = (
            mlp if mlp is not None else Cohere2MLP(embed_dim, mlp_dim, name="mlp")
        )

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
                "layer_type": self.layer_type,
                "norm_eps": self.norm_eps,
                "attention_bias": self.attention_bias,
            }
        )
        return config
