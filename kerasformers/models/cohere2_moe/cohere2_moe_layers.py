import keras
from keras import layers, ops

from kerasformers.models.cohere2.cohere2_layers import (
    Cohere2Attention,
    Cohere2LayerNorm,
)


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2MoeRMSNorm(layers.Layer):
    """Root-mean-square norm (used when the checkpoint sets ``rms_norm_eps``).

    Scales by ``rsqrt(mean(x**2) + eps)`` then by a learned ones-init weight —
    no mean-centering, unlike Cohere's :class:`~kerasformers.models.cohere2.cohere2_layers.Cohere2LayerNorm`.
    The reduction runs in float32 and casts back to the input dtype. RMSNorm
    Cohere2-MoE checkpoints (e.g. North-Mini) select this through ``make_norm``.

    Args:
        eps: Variance epsilon. Defaults to ``1e-5``.
    """

    def __init__(self, eps=1e-5, **kwargs):
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
        return ops.cast(ops.cast(self.weight, "float32") * x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


def make_norm(use_rms_norm, eps, name):
    """Build the normalization layer a Cohere2-MoE checkpoint asks for.

    Args:
        use_rms_norm: When ``True`` return a :class:`Cohere2MoeRMSNorm`; otherwise
            a mean-centered :class:`~kerasformers.models.cohere2.cohere2_layers.Cohere2LayerNorm`.
        eps: Variance epsilon passed to the norm.
        name: Layer name.

    Returns:
        The constructed normalization layer.
    """
    if use_rms_norm:
        return Cohere2MoeRMSNorm(eps=eps, name=name)
    return Cohere2LayerNorm(eps=eps, name=name)


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2MoeMLP(layers.Layer):
    """Cohere2-MoE dense / shared-expert SwiGLU MLP: ``down(silu(gate(x)) * up(x))``.

    Bias-free SwiGLU used for the leading dense layer(s) and the shared expert.

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
class Cohere2MoeExperts(layers.Layer):
    """Dense bank of SwiGLU experts evaluated with fused einsums (HF layout).

    Holds every expert in two stacked weights — ``gate_up_proj``
    ``(num_experts, 2*mlp_dim, embed_dim)`` and ``down_proj``
    ``(num_experts, embed_dim, mlp_dim)`` — and runs all experts for all tokens
    in three einsums, weighting each expert's output by the per-token routing
    weights (zero-weighted experts drop out of the sum). Computing every expert
    for every token is wasteful but vectorizes cleanly across backends.

    Args:
        num_experts: Number of routed experts.
        embed_dim: Model width (expert input/output dimension).
        mlp_dim: Per-expert SwiGLU hidden width.

    Call args:
        hidden_states: ``(num_tokens, embed_dim)`` flattened token rows.
        routing_weights: ``(num_tokens, num_experts)`` dense routing weights
            (zero for unselected experts).

    Returns:
        ``(num_tokens, embed_dim)`` combined expert output.
    """

    def __init__(self, num_experts, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        self.gate_up_proj = self.add_weight(
            name="gate_up_proj",
            shape=(self.num_experts, 2 * self.mlp_dim, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.down_proj = self.add_weight(
            name="down_proj",
            shape=(self.num_experts, self.embed_dim, self.mlp_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states, routing_weights):
        gate_up = ops.einsum("th,eoh->teo", hidden_states, self.gate_up_proj)
        gate = gate_up[..., : self.mlp_dim]
        up = gate_up[..., self.mlp_dim :]
        expert_out = ops.einsum("tei,ehi->teh", ops.silu(gate) * up, self.down_proj)
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
class Cohere2MoeSparseBlock(layers.Layer):
    """Cohere2-MoE block: top-k-FIRST router, then softmax/sigmoid + shared experts.

    Distinct from Qwen/Mixtral: the router takes the top-``num_experts_per_tok``
    raw logits *first*, then normalizes only those — ``softmax`` over the k
    (no ``norm_topk_prob``), or ``sigmoid`` (with optional ``norm_topk_prob``).
    Optional shared expert(s) combined by ``sum`` or ``average``.

    Args:
        num_experts / num_experts_per_tok: Routing shape.
        embed_dim / moe_mlp_dim: Expert dims.
        expert_selection_fn: ``"softmax"`` or ``"sigmoid"``.
        norm_topk_prob: Renormalize the selected weights (sigmoid only).
        num_shared_experts: Shared-expert count (0 disables).
        shared_mlp_dim: Shared-expert hidden width.
        shared_combine: ``"sum"`` or ``"average"``.
    """

    def __init__(
        self,
        num_experts,
        num_experts_per_tok,
        embed_dim,
        moe_mlp_dim,
        expert_selection_fn="softmax",
        norm_topk_prob=True,
        num_shared_experts=0,
        shared_mlp_dim=0,
        shared_combine="average",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.moe_mlp_dim = moe_mlp_dim
        self.expert_selection_fn = expert_selection_fn
        self.norm_topk_prob = norm_topk_prob
        self.num_shared_experts = num_shared_experts
        self.shared_mlp_dim = shared_mlp_dim
        self.shared_combine = shared_combine
        self.experts = Cohere2MoeExperts(
            num_experts, embed_dim, moe_mlp_dim, name="experts"
        )
        if num_shared_experts > 0:
            self.shared_experts = Cohere2MoeMLP(
                embed_dim, shared_mlp_dim, name="shared_experts"
            )

    def build(self, input_shape):
        self.gate_weight = self.add_weight(
            name="gate_weight",
            shape=(self.num_experts, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        x = ops.reshape(hidden_states, (-1, self.embed_dim))
        logits = ops.matmul(x, ops.transpose(self.gate_weight))
        top_logits, top_idx = ops.top_k(logits, self.num_experts_per_tok)
        if self.expert_selection_fn == "softmax":
            top_w = ops.softmax(ops.cast(top_logits, "float32"), axis=-1)
        else:
            top_w = ops.sigmoid(ops.cast(top_logits, "float32"))
            if self.norm_topk_prob:
                top_w = top_w / ops.sum(top_w, axis=-1, keepdims=True)
        top_w = ops.cast(top_w, x.dtype)
        one_hot = ops.one_hot(top_idx, self.num_experts, dtype=x.dtype)
        routing = ops.sum(one_hot * top_w[..., None], axis=1)
        out = self.experts(x, routing)
        if self.num_shared_experts > 0:
            shared = self.shared_experts(x)
            out = out + shared if self.shared_combine == "sum" else (out + shared) / 2
        return ops.reshape(out, (b, s, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "embed_dim": self.embed_dim,
                "moe_mlp_dim": self.moe_mlp_dim,
                "expert_selection_fn": self.expert_selection_fn,
                "norm_topk_prob": self.norm_topk_prob,
                "num_shared_experts": self.num_shared_experts,
                "shared_mlp_dim": self.shared_mlp_dim,
                "shared_combine": self.shared_combine,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2MoeDecoderLayer(layers.Layer):
    """One Cohere2-MoE block: parallel attention + (dense MLP | MoE).

    ``h = x + attention(norm(x)) + mlp(norm(x))`` — attention and the
    feed-forward both read the single input norm and add into the residual.
    ``use_rope`` controls the NoPE-vs-rope decision (sliding layers, and the
    force-rope dense prefix layers, get rope; full layers run NoPE) and
    ``use_moe`` swaps the dense :class:`Cohere2MoeMLP` for a
    :class:`Cohere2MoeSparseBlock`. The shared input norm is
    :class:`Cohere2MoeRMSNorm` or mean-centered Cohere LayerNorm per
    ``use_rms_norm``.

    Args:
        embed_dim / num_heads / num_kv_heads / head_dim: Attention dims.
        layer_type: ``"sliding_attention"`` or ``"full_attention"``.
        use_rope: Apply rotary embeddings (sliding / force-rope layers) or NoPE.
        use_moe: Use the sparse expert block instead of a dense MLP.
        dense_mlp_dim: Hidden width when ``use_moe`` is ``False``.
        num_experts / num_experts_per_tok / moe_mlp_dim: Routed-expert shape.
        expert_selection_fn / norm_topk_prob: Router knobs (see
            :class:`Cohere2MoeSparseBlock`).
        num_shared_experts / shared_mlp_dim / shared_combine: Shared-expert config.
        use_rms_norm: Pick RMSNorm vs Cohere LayerNorm for the input norm.
        norm_eps: Norm epsilon.
        attention_bias: Attention projection bias.

    Call args:
        hidden_states: ``(batch, seq_len, embed_dim)``.
        cos, sin: interleaved rotary tables (used only when ``use_rope``).
        attention_mask: additive mask for this layer's ``layer_type``, or ``None``.
        use_cache: when ``True``, also return the attention ``(key, value)``.

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
        layer_type,
        use_rope,
        use_moe,
        dense_mlp_dim,
        num_experts,
        num_experts_per_tok,
        moe_mlp_dim,
        expert_selection_fn,
        norm_topk_prob,
        num_shared_experts,
        shared_mlp_dim,
        shared_combine,
        use_rms_norm=False,
        norm_eps=1e-5,
        attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_type = layer_type
        self.use_rope = use_rope
        self.use_moe = use_moe
        self.dense_mlp_dim = dense_mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.expert_selection_fn = expert_selection_fn
        self.norm_topk_prob = norm_topk_prob
        self.num_shared_experts = num_shared_experts
        self.shared_mlp_dim = shared_mlp_dim
        self.shared_combine = shared_combine
        self.use_rms_norm = use_rms_norm
        self.norm_eps = norm_eps
        self.attention_bias = attention_bias
        self.input_layernorm = make_norm(use_rms_norm, norm_eps, "input_layernorm")
        self.attention = Cohere2Attention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            use_rope=use_rope,
            attention_bias=attention_bias,
            name="attention",
        )
        if use_moe:
            self.mlp = Cohere2MoeSparseBlock(
                num_experts,
                num_experts_per_tok,
                embed_dim,
                moe_mlp_dim,
                expert_selection_fn,
                norm_topk_prob,
                num_shared_experts,
                shared_mlp_dim,
                shared_combine,
                name="mlp",
            )
        else:
            self.mlp = Cohere2MoeMLP(embed_dim, dense_mlp_dim, name="mlp")

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
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "layer_type": self.layer_type,
                "use_rope": self.use_rope,
                "use_moe": self.use_moe,
                "dense_mlp_dim": self.dense_mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "expert_selection_fn": self.expert_selection_fn,
                "norm_topk_prob": self.norm_topk_prob,
                "num_shared_experts": self.num_shared_experts,
                "shared_mlp_dim": self.shared_mlp_dim,
                "shared_combine": self.shared_combine,
                "use_rms_norm": self.use_rms_norm,
                "norm_eps": self.norm_eps,
                "attention_bias": self.attention_bias,
            }
        )
        return config
