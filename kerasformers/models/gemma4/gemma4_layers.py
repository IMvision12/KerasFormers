import keras
from keras import layers, ops


def apply_rope(x, cos, sin):
    # Full-width half-rotation rope on (B, L, H, D); partial ("proportional")
    # rotary is realized upstream by zero-padding the inverse frequencies to
    # head_dim // 2 (cos(0) = 1, sin(0) = 0 leaves those dims untouched),
    # exactly like HF.
    cos = ops.expand_dims(cos, axis=2)
    sin = ops.expand_dims(sin, axis=2)
    half = ops.shape(x)[-1] // 2
    rot = ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4RMSNorm(layers.Layer):
    """Gemma 4 root-mean-square layer norm — unlike earlier Gemmas, a *plain*
    ``* weight`` scale (ones-initialized), optionally weightless
    (``with_scale=False`` — the value norm and the router input norm carry no
    checkpoint parameters).

    Args:
        eps: Variance epsilon. Defaults to ``1e-6``.
        with_scale: Whether a learned scale is applied.
    """

    def __init__(self, eps=1e-6, with_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.with_scale = with_scale

    def build(self, input_shape):
        if self.with_scale:
            self.weight = self.add_weight(
                name="weight",
                shape=(input_shape[-1],),
                initializer="ones",
                trainable=True,
            )
        self.built = True

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(variance + self.eps)
        if self.with_scale:
            x = x * ops.cast(self.weight, "float32")
        return ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps, "with_scale": self.with_scale})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4MLP(layers.Layer):
    """Gemma 4 GeGLU feed-forward block: ``down(gelu_tanh(gate(x)) * up(x))``.

    Bias-free GeGLU: the ``gate`` branch uses the tanh ``gelu`` approximation,
    is multiplied elementwise by the ``up`` projection, and ``down`` projects
    the result back to ``embed_dim``.

    Args:
        embed_dim: Model width (input and output dimension).
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
        return self.down(ops.gelu(self.gate(x), approximate=True) * self.up(x))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4Experts(layers.Layer):
    """Gemma 4 fused routed-expert bank (dense evaluation, GeGLU experts).

    Hugging Face fused layout — ``gate_up_proj`` ``(E, 2I, H)`` (contiguous
    halves), ``down_proj`` ``(E, H, I)``, no biases — with the
    ``gelu_pytorch_tanh`` activation. Given per-token per-expert routing
    weights ``(T, E)``, computes every expert and combines the outputs.

    Args:
        num_experts: Number of routed experts ``E``.
        embed_dim: Model width ``H``.
        mlp_dim: Per-expert hidden width ``I``.
    """

    def __init__(self, num_experts, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        e, h, i = self.num_experts, self.embed_dim, self.mlp_dim
        self.gate_up_proj = self.add_weight(
            name="gate_up_proj",
            shape=(e, 2 * i, h),
            initializer="zeros",
            trainable=True,
        )
        self.down_proj = self.add_weight(
            name="down_proj", shape=(e, h, i), initializer="zeros", trainable=True
        )
        self.built = True

    def call(self, hidden_states, routing_weights):
        gate_up = ops.einsum("th,eoh->teo", hidden_states, self.gate_up_proj)
        gate = gate_up[..., : self.mlp_dim]
        up = gate_up[..., self.mlp_dim :]
        act = ops.gelu(gate, approximate=True) * up
        expert_out = ops.einsum("tei,ehi->teh", act, self.down_proj)
        return ops.einsum("te,teh->th", routing_weights, expert_out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4Router(layers.Layer):
    """Gemma 4 expert router.

    The input is RMS-normalized (weightless), scaled by a learned per-channel
    ``scale`` times ``hidden**-0.5``, projected to expert logits, softmaxed;
    the top-k weights are renormalized and multiplied by a learned
    ``per_expert_scale``. Returns dense ``(T, E)`` routing weights.

    Args:
        num_experts: Routed expert count.
        top_k: Experts kept per token.
        embed_dim: Model width.
        norm_eps: Epsilon of the input norm.
    """

    def __init__(self, num_experts, top_k, embed_dim, norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.embed_dim = embed_dim
        self.norm_eps = norm_eps
        self.norm = Gemma4RMSNorm(eps=norm_eps, with_scale=False, name="norm")
        self.proj = layers.Dense(num_experts, use_bias=False, name="proj")

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", shape=(self.embed_dim,), initializer="ones", trainable=True
        )
        self.per_expert_scale = self.add_weight(
            name="per_expert_scale",
            shape=(self.num_experts,),
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def call(self, x):
        x = self.norm(x)
        x = x * self.scale * ops.cast(self.embed_dim**-0.5, x.dtype)
        probs = ops.softmax(self.proj(x), axis=-1)  # (T, E)
        top_vals, top_idx = ops.top_k(probs, self.top_k)
        top_vals = top_vals / ops.sum(top_vals, axis=-1, keepdims=True)
        top_vals = top_vals * ops.take(self.per_expert_scale, top_idx)
        one_hot = ops.one_hot(top_idx, self.num_experts)
        return ops.sum(one_hot * top_vals[..., None], axis=1)  # (T, E)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "embed_dim": self.embed_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4Attention(layers.Layer):
    """Gemma 4 self-attention with per-layer geometry and K=V global layers.

    Sliding layers: ``head_dim`` (256), ``num_kv_heads`` K/V heads, separate
    value projection, full-width default rope (theta 1e4). Global layers:
    ``global_head_dim`` (512), ``num_global_kv_heads`` (MQA-ish) and — when
    ``k_eq_v`` — *no value projection*: the value is the raw key projection,
    normalized by a weightless RMS norm. Per-head q/k ``(1 + w)`` RMS norms
    are applied before rope; global layers rotate only the first
    ``partial_rotary_factor`` fraction of the head ("proportional" rope,
    theta 1e6). Attention scores are unscaled (``scaling = 1.0``).

    Args:
        embed_dim: Model width.
        num_heads: Query heads.
        num_kv_heads: K/V heads for this layer.
        head_dim: Per-head dim for this layer.
        k_eq_v: Whether value = normalized key projection (global layers).
        norm_eps: Epsilon of the q/k/v norms.

    Call args:
        hidden_states, cos, sin (width = rotated dims), attention_mask,
        past_key_value, use_cache: standard decoder-attention arguments.

    Returns:
        Output ``(batch, q_len, embed_dim)``, or ``(output, (key, value))``
        when ``use_cache`` is set.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        k_eq_v=False,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k_eq_v = k_eq_v
        self.norm_eps = norm_eps
        self.num_kv_groups = num_heads // num_kv_heads
        self.query = layers.Dense(num_heads * head_dim, use_bias=False, name="query")
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = (
            None
            if k_eq_v
            else layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        self.query_norm = Gemma4RMSNorm(eps=norm_eps, name="query_norm")
        self.key_norm = Gemma4RMSNorm(eps=norm_eps, name="key_norm")
        self.value_norm = Gemma4RMSNorm(
            eps=norm_eps, with_scale=False, name="value_norm"
        )

    def project_qkv(self, hidden_states, q_len, cos, sin):
        b = ops.shape(hidden_states)[0]
        q = ops.reshape(
            self.query(hidden_states), (b, q_len, self.num_heads, self.head_dim)
        )
        q = self.query_norm(q)
        q = apply_rope(q, cos, sin)
        q = ops.transpose(q, (0, 2, 1, 3))

        k_raw = ops.reshape(
            self.key(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        if self.value is not None:
            v = ops.reshape(
                self.value(hidden_states),
                (b, q_len, self.num_kv_heads, self.head_dim),
            )
        else:
            v = k_raw
        k = self.key_norm(k_raw)
        k = apply_rope(k, cos, sin)
        k = ops.transpose(k, (0, 2, 1, 3))
        v = self.value_norm(v)
        v = ops.transpose(v, (0, 2, 1, 3))
        return q, k, v

    def attend(self, q, k, v, attention_mask, b, q_len):
        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))  # scaling = 1.0
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, v)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, q_len, self.num_heads * self.head_dim)
        )
        return self.output_proj(out)

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
        q, k, v = self.project_qkv(hidden_states, q_len, cos, sin)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = ops.concatenate([past_k, k], axis=2)
            v = ops.concatenate([past_v, v], axis=2)
        new_kv = (k, v) if use_cache else None
        out = self.attend(q, k, v, attention_mask, b, q_len)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        # Single-token attention against fixed-size caches.
        b = ops.shape(hidden_states)[0]
        q, k, v = self.project_qkv(hidden_states, 1, cos, sin)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        kk, vv = cache_k, cache_v
        if self.num_kv_groups > 1:
            kk = ops.repeat(kk, self.num_kv_groups, axis=1)
            vv = ops.repeat(vv, self.num_kv_groups, axis=1)
        attn = ops.matmul(q, ops.transpose(kk, (0, 1, 3, 2)))
        attn = attn + key_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, vv)
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
                "k_eq_v": self.k_eq_v,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4DecoderLayer(layers.Layer):
    """One Gemma 4 text block: four-norm sandwich, optional parallel MoE
    branch, and a learned ``layer_scalar`` output multiplier.

    Dense layers: ``h = res + post_ff(mlp(pre_ff(h)))``. MoE layers (26B-A4B):
    the dense branch is normed (``post_ff_1``), a routed-expert branch is
    computed from the *residual* (``pre_ff_2`` -> experts -> ``post_ff_2``),
    the two are summed, then ``post_ff`` + residual as usual.

    Args:
        embed_dim: Model / residual-stream width.
        mlp_dim: Dense GeGLU hidden width.
        num_heads: Query heads.
        num_kv_heads: K/V heads for this layer.
        head_dim: Per-head dim for this layer.
        k_eq_v: Whether the attention is the global K=V kind.
        is_moe: Whether this layer carries the parallel expert branch.
        num_experts / num_experts_per_tok / moe_mlp_dim: MoE parameters.
        norm_eps: Epsilon of all norms.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        k_eq_v=False,
        is_moe=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_mlp_dim=0,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k_eq_v = k_eq_v
        self.is_moe = is_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.norm_eps = norm_eps
        self.attention_norm = Gemma4RMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Gemma4Attention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            k_eq_v,
            norm_eps,
            name="attention",
        )
        self.post_attention_norm = Gemma4RMSNorm(
            eps=norm_eps, name="post_attention_norm"
        )
        self.pre_feedforward_norm = Gemma4RMSNorm(
            eps=norm_eps, name="pre_feedforward_norm"
        )
        self.mlp = Gemma4MLP(embed_dim, mlp_dim, name="mlp")
        self.post_feedforward_norm = Gemma4RMSNorm(
            eps=norm_eps, name="post_feedforward_norm"
        )
        if is_moe:
            self.router = Gemma4Router(
                num_experts, num_experts_per_tok, embed_dim, norm_eps, name="router"
            )
            self.experts = Gemma4Experts(
                num_experts, embed_dim, moe_mlp_dim, name="experts"
            )
            self.post_feedforward_norm_1 = Gemma4RMSNorm(
                eps=norm_eps, name="post_feedforward_norm_1"
            )
            self.pre_feedforward_norm_2 = Gemma4RMSNorm(
                eps=norm_eps, name="pre_feedforward_norm_2"
            )
            self.post_feedforward_norm_2 = Gemma4RMSNorm(
                eps=norm_eps, name="post_feedforward_norm_2"
            )

    def build(self, input_shape):
        self.layer_scalar = self.add_weight(
            name="layer_scalar", shape=(1,), initializer="ones", trainable=True
        )
        self.built = True

    def feed_forward(self, residual):
        h = self.pre_feedforward_norm(residual)
        h = self.mlp(h)
        if self.is_moe:
            h1 = self.post_feedforward_norm_1(h)
            flat = ops.reshape(residual, (-1, self.embed_dim))
            routing = self.router(flat)
            h2 = self.pre_feedforward_norm_2(flat)
            h2 = self.experts(h2, ops.cast(routing, h2.dtype))
            h2 = ops.reshape(h2, ops.shape(residual))
            h2 = self.post_feedforward_norm_2(h2)
            h = h1 + h2
        h = self.post_feedforward_norm(h)
        return residual + h

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
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = residual + self.post_attention_norm(attn_out)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = hidden_states * ops.cast(self.layer_scalar, hidden_states.dtype)
        return (hidden_states, new_kv) if use_cache else hidden_states

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        attn_out, cache_k, cache_v = self.attention.decode_step(
            x, cos, sin, cache_k, cache_v, write_pos, key_mask
        )
        hidden_states = residual + self.post_attention_norm(attn_out)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = hidden_states * ops.cast(self.layer_scalar, hidden_states.dtype)
        return hidden_states, cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "k_eq_v": self.k_eq_v,
                "is_moe": self.is_moe,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config
