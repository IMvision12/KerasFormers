import keras
from keras import layers, ops

from kerasformers.base.base_attention import fused_attention


def apply_cohere_rope(x, cos, sin):
    """Apply Cohere's interleaved rotary position embedding to ``x``.

    Cohere rotates adjacent channel pairs ``(x[2i], x[2i+1])`` by a shared
    angle: ``rotate_half`` is ``[-x1, x0, -x3, x2, ...]`` (each pair becomes
    ``(-odd, even)``), and ``cos`` / ``sin`` are repeat-interleaved so the same
    angle covers both members of a pair. The rotation runs in float32 to match
    the reference implementation, then casts back to ``x``'s dtype.

    Args:
        x: Query or key tensor, shape ``(batch, num_heads, seq_len, head_dim)``.
        cos: Cosine table, repeat-interleaved along the last axis and
            broadcastable to ``x`` (e.g. ``(batch, 1, seq_len, head_dim)``).
        sin: Sine table, same shape/layout as ``cos``.

    Returns:
        The rotary-embedded tensor, same shape and dtype as ``x``.
    """
    dtype = x.dtype
    x = ops.cast(x, "float32")
    cos = ops.cast(cos, "float32")
    sin = ops.cast(sin, "float32")
    rotate_half = ops.reshape(
        ops.stack([-x[..., 1::2], x[..., 0::2]], axis=-1), ops.shape(x)
    )
    out = x * cos + rotate_half * sin
    return ops.cast(out, dtype)


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2LayerNorm(layers.Layer):
    """Cohere2 LayerNorm: mean-centered, bias-free, ones-init weight.

    Normalizes over the last axis in float32 as ``(x - mean) * rsqrt(var + eps)``
    then scales by a learned weight (no bias). Used for the input and final
    norms (Cohere2 has no QK-norm).

    Args:
        eps: Variance epsilon. Defaults to ``1e-5``.
        weight_shape: Explicit scale-weight shape; ``None`` (default) infers
            ``(input_dim,)`` from the input.
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
class Cohere2MLP(layers.Layer):
    """Cohere2 SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))`` (bias-free).

    Args:
        embed_dim: Model width (the input and output dimension).
        mlp_dim: Hidden width of the ``gate`` / ``up`` projections.
    """

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
class Cohere2Attention(layers.Layer):
    """Cohere2 grouped-query attention: NoPE on full layers, rope on sliding.

    No QK-norm. Rotary (Cohere interleaved) is applied **only** on the sliding-
    window layers; the full-attention layers run without positional encoding
    (NoPE). The model passes the matching mask (sliding or full causal).

    Args:
        embed_dim / num_heads / num_kv_heads / head_dim: Geometry (grouped-query
            attention when ``num_kv_heads < num_heads``; K/V heads are repeated).
        use_rope: Apply rotary (True on sliding layers, False on full/NoPE).
        attention_bias: Whether the projections carry bias.

    Call args:
        hidden_states: ``(batch, seq_len, embed_dim)``.
        cos, sin: interleaved rotary tables ``(batch, seq_len, head_dim)``
            (ignored when ``use_rope`` is False).
        attention_mask: additive mask broadcastable to
            ``(batch, 1, q_len, kv_len)`` (sliding-window or full causal), or
            ``None``.
        use_cache: when ``True``, also return the new ``(key, value)``.

    Returns:
        ``(batch, seq_len, embed_dim)``, or ``(out, (key, value))`` when
        ``use_cache``. :meth:`decode_step` runs a single cached decode step.
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
        return fused_attention(q, k, v, self.scaling, attention_mask)

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
        mlp: Optional MLP/MoE instance override (used by Cohere2-MoE to swap in
            the sparse block); defaults to a :class:`Cohere2MLP`.

    Call args:
        hidden_states: ``(batch, seq_len, embed_dim)``.
        cos, sin: interleaved rotary tables (used only on sliding layers).
        attention_mask: additive mask for this layer's ``layer_type``, or ``None``.
        use_cache: when ``True``, also return the attention ``(key, value)``.

    Returns:
        ``(batch, seq_len, embed_dim)``, or ``(out, (key, value))`` when
        ``use_cache``. :meth:`decode_step` runs a single cached decode step.
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
