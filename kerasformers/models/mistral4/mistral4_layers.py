import keras
from keras import layers, ops


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def deinterleave_pairs(x):
    # Mistral Large 3 checkpoints store the rope dims interleaved
    # (re0, im0, re1, im1, ...); regroup them to the half layout
    # (re0, re1, ..., im0, im1, ...) so the standard half-rotation applies
    # (HF apply_rotary_pos_emb_interleave).
    b = ops.shape(x)[0]
    h = ops.shape(x)[1]
    s = ops.shape(x)[2]
    d = ops.shape(x)[-1]
    x = ops.reshape(x, (b, h, s, d // 2, 2))
    x = ops.transpose(x, (0, 1, 2, 4, 3))
    return ops.reshape(x, (b, h, s, d))


@keras.saving.register_keras_serializable(package="kerasformers")
class Mistral4RMSNorm(layers.Layer):
    """Root-mean-square layer norm (Mistral Large 3 style).

    Normalizes the last axis by its RMS in float32 (for numerical stability),
    casts back to the input dtype, then scales by a learned per-channel weight.
    No mean subtraction, no bias. Shape-preserving: ``(..., dim) -> (..., dim)``.

    Args:
        eps: Variance epsilon added before the reciprocal square root.
            Defaults to ``1e-6``.
    """

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
class Mistral4MLP(layers.Layer):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``, bias-free.

    Used for the first ``first_k_dense_replace`` dense layers (width
    ``mlp_dim_dense``) and as every MoE layer's always-active shared expert
    (width ``moe_mlp_dim * n_shared_experts``).
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
class Mistral4Experts(layers.Layer):
    """Mistral Large 3 fused routed-expert bank (dense evaluation).

    Per-expert SwiGLU parameters in Hugging Face's fused layout —
    ``gate_up_proj`` ``(E, 2I, H)`` (contiguous gate/up halves), ``down_proj``
    ``(E, H, I)``, no biases. Given per-token per-expert routing weights
    ``(T, E)`` (zero for non-selected experts), computes every expert and
    combines the outputs by those weights — identical to sparse top-k routing
    with compute O(num_experts).

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
class Mistral4MoE(layers.Layer):
    """Mistral Large 3 mixture of experts (DeepSeek-V3-style routing).

    The bias-free router's logits are softmaxed over all experts in float32;
    experts are organized in ``n_group`` groups — each group is scored by the
    sum of its top-2 probabilities, the best ``topk_group`` groups survive,
    and the final top-``num_experts_per_tok`` experts are picked from the
    surviving groups (the released model uses a single group, collapsing this
    to plain top-k). Kept weights are renormalized when ``norm_topk_prob``
    and scaled by ``routed_scaling_factor``; expert outputs are combined by
    those weights and an always-active shared expert is added.

    Args:
        num_experts: Routed expert count.
        num_experts_per_tok: Top-k experts per token.
        embed_dim: Model width.
        mlp_dim: Per-expert hidden width (``moe_intermediate_size``).
        n_shared_experts: Shared-expert width multiplier.
        n_group / topk_group: Expert-group routing parameters.
        norm_topk_prob: Renormalize the kept top-k weights to sum to one.
        routed_scaling_factor: Final scale on the routing weights.
    """

    def __init__(
        self,
        num_experts,
        num_experts_per_tok,
        embed_dim,
        mlp_dim,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.experts = Mistral4Experts(num_experts, embed_dim, mlp_dim, name="experts")
        self.shared_experts = Mistral4MLP(
            embed_dim, mlp_dim * n_shared_experts, name="shared_experts"
        )

    def build(self, input_shape):
        self.router_weight = self.add_weight(
            name="router_weight",
            shape=(self.num_experts, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def route(self, x):
        probs = ops.softmax(
            ops.cast(ops.matmul(x, ops.transpose(self.router_weight)), "float32"),
            axis=-1,
        )  # (T, E)
        if self.n_group > 1:
            grouped = ops.reshape(
                probs, (-1, self.n_group, self.num_experts // self.n_group)
            )
            group_scores = ops.sum(ops.top_k(grouped, 2)[0], axis=-1)  # (T, G)
            _, group_idx = ops.top_k(group_scores, self.topk_group)
            group_mask = ops.sum(ops.one_hot(group_idx, self.n_group), axis=1)  # (T, G)
            score_mask = ops.reshape(
                ops.broadcast_to(
                    group_mask[..., None],
                    (
                        ops.shape(group_mask)[0],
                        self.n_group,
                        self.num_experts // self.n_group,
                    ),
                ),
                (-1, self.num_experts),
            )
            choice = probs * score_mask
        else:
            choice = probs
        _, top_idx = ops.top_k(choice, self.num_experts_per_tok)
        one_hot = ops.one_hot(top_idx, self.num_experts)  # (T, k, E)
        selected = ops.sum(one_hot, axis=1)  # (T, E)
        top_weights = probs * selected
        if self.norm_topk_prob:
            top_weights = top_weights / (
                ops.sum(top_weights, axis=-1, keepdims=True) + 1e-20
            )
        return top_weights * self.routed_scaling_factor

    def call(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        x = ops.reshape(hidden_states, (-1, self.embed_dim))
        routing = ops.cast(self.route(x), x.dtype)
        out = self.experts(x, routing) + self.shared_experts(x)
        return ops.reshape(out, (b, s, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "n_shared_experts": self.n_shared_experts,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "norm_topk_prob": self.norm_topk_prob,
                "routed_scaling_factor": self.routed_scaling_factor,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Mistral4Attention(layers.Layer):
    """Multi-head latent attention (MLA) with low-rank q/kv compression.

    The query path is compressed (``q_a_proj`` to ``q_lora_rank``, RMS-normed,
    re-expanded by ``q_b_proj``); keys/values share a compressed latent
    (``kv_a_proj_with_mqa`` to ``kv_lora_rank`` + a head-shared
    ``qk_rope_head_dim`` rotary slice, RMS-normed, expanded by ``kv_b_proj``).
    Per head, queries/keys are the concatenation of a ``qk_nope_head_dim``
    non-rotary part and a ``qk_rope_head_dim`` rotary part (the checkpoint
    stores the rotary dims interleaved — they are regrouped before the
    half-rotation). Values use ``v_head_dim``. Queries are additionally
    scaled by the position-dependent Llama-4 attention temperature
    ``1 + beta * log(1 + floor(pos / original_max))`` when ``llama4_beta``
    is set.

    Args:
        embed_dim: Model width.
        num_heads: Attention heads.
        q_lora_rank: Query compression rank (``None`` disables, using a
            direct ``q_proj``).
        kv_lora_rank: Key/value compression rank.
        qk_nope_head_dim / qk_rope_head_dim: Non-rotary / rotary per-head
            query-key dims.
        v_head_dim: Per-head value dim.
        norm_eps: Epsilon of the latent RMS norms.
        rope_interleave: Whether the rotary dims are stored interleaved.
        llama4_beta: Temperature-scaling beta (``None`` disables).
        llama4_original_max: Position threshold of the temperature scaling.

    Call args:
        hidden_states: ``(batch, q_len, embed_dim)``.
        cos, sin: rotary tables ``(batch, q_len, qk_rope_head_dim)``.
        positions: ``(batch, q_len)`` int positions (temperature scaling).
        attention_mask: additive mask, or ``None``.
        past_key_value: optional ``(past_k, past_v)`` with
            ``(batch, heads, len, qk_head)`` / ``(batch, heads, len, v_head)``.
        use_cache: when ``True``, also return the updated ``(key, value)``.

    Returns:
        Output ``(batch, q_len, embed_dim)``, or ``(output, (key, value))``
        when ``use_cache`` is set.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        norm_eps=1e-6,
        rope_interleave=True,
        llama4_beta=0.1,
        llama4_original_max=8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.norm_eps = norm_eps
        self.rope_interleave = rope_interleave
        self.llama4_beta = llama4_beta
        self.llama4_original_max = llama4_original_max
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.scaling = self.qk_head_dim**-0.5

        if q_lora_rank is None:
            self.q_proj = layers.Dense(
                num_heads * self.qk_head_dim, use_bias=False, name="q_proj"
            )
        else:
            self.q_a_proj = layers.Dense(q_lora_rank, use_bias=False, name="q_a_proj")
            self.q_a_layernorm = Mistral4RMSNorm(eps=norm_eps, name="q_a_layernorm")
            self.q_b_proj = layers.Dense(
                num_heads * self.qk_head_dim, use_bias=False, name="q_b_proj"
            )
        self.kv_a_proj_with_mqa = layers.Dense(
            kv_lora_rank + qk_rope_head_dim, use_bias=False, name="kv_a_proj_with_mqa"
        )
        self.kv_a_layernorm = Mistral4RMSNorm(eps=norm_eps, name="kv_a_layernorm")
        self.kv_b_proj = layers.Dense(
            num_heads * (qk_nope_head_dim + v_head_dim),
            use_bias=False,
            name="kv_b_proj",
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def project_qkv(self, hidden_states, q_len, cos, sin, positions):
        b = ops.shape(hidden_states)[0]
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = ops.transpose(
            ops.reshape(q, (b, q_len, self.num_heads, self.qk_head_dim)), (0, 2, 1, 3)
        )
        q_nope = q[..., : self.qk_nope_head_dim]
        q_rot = q[..., self.qk_nope_head_dim :]

        compressed = self.kv_a_proj_with_mqa(hidden_states)
        k_latent = compressed[..., : self.kv_lora_rank]
        k_rot = compressed[..., self.kv_lora_rank :]
        kv = self.kv_b_proj(self.kv_a_layernorm(k_latent))
        kv = ops.transpose(
            ops.reshape(
                kv, (b, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            ),
            (0, 2, 1, 3),
        )
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k_rot = ops.reshape(k_rot, (b, 1, q_len, self.qk_rope_head_dim))

        if self.rope_interleave:
            q_rot = deinterleave_pairs(q_rot)
            k_rot = deinterleave_pairs(k_rot)
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q_rot = q_rot * cos + rotate_half(q_rot) * sin
        k_rot = k_rot * cos + rotate_half(k_rot) * sin
        k_rot = ops.broadcast_to(
            k_rot, (b, self.num_heads, q_len, self.qk_rope_head_dim)
        )

        q = ops.concatenate([q_nope, q_rot], axis=-1)
        k = ops.concatenate([k_nope, k_rot], axis=-1)
        if self.llama4_beta is not None:
            pos = ops.cast(positions, "float32")
            scale = 1.0 + self.llama4_beta * ops.log(
                1.0 + ops.floor(pos / self.llama4_original_max)
            )
            q = ops.cast(ops.cast(q, "float32") * scale[:, None, :, None], q.dtype)
        return q, k, v

    def attend(self, q, k, v, attention_mask, b, q_len):
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, v)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)),
            (b, q_len, self.num_heads * self.v_head_dim),
        )
        return self.output_proj(out)

    def call(
        self,
        hidden_states,
        cos,
        sin,
        positions,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]
        q, k, v = self.project_qkv(hidden_states, q_len, cos, sin, positions)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = ops.concatenate([past_k, k], axis=2)
            v = ops.concatenate([past_v, v], axis=2)
        new_kv = (k, v) if use_cache else None
        out = self.attend(q, k, v, attention_mask, b, q_len)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self,
        hidden_states,
        cos,
        sin,
        positions,
        cache_k,
        cache_v,
        write_pos,
        key_mask,
    ):
        # Single-token MLA step against fixed-size caches (k: qk_head_dim,
        # v: v_head_dim slots).
        b = ops.shape(hidden_states)[0]
        q, k, v = self.project_qkv(hidden_states, 1, cos, sin, positions)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        attn = ops.matmul(q, ops.transpose(cache_k, (0, 1, 3, 2))) * self.scaling
        attn = attn + key_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, cache_v)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, 1, self.num_heads * self.v_head_dim)
        )
        return self.output_proj(out), cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "norm_eps": self.norm_eps,
                "rope_interleave": self.rope_interleave,
                "llama4_beta": self.llama4_beta,
                "llama4_original_max": self.llama4_original_max,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Mistral4DecoderLayer(layers.Layer):
    """One Mistral Large 3 block: pre-norm MLA attention, then pre-norm
    feed-forward (dense SwiGLU on the first ``first_k_dense_replace`` layers,
    the shared-expert MoE elsewhere).

    Args:
        embed_dim: Model / residual-stream width.
        mlp_dim_dense: Dense layers' SwiGLU hidden width.
        moe_mlp_dim: Per-routed-expert hidden width.
        num_heads: Attention heads.
        is_moe: Whether this layer's feed-forward is the MoE block.
        num_experts / num_experts_per_tok / n_shared_experts / n_group /
        topk_group / norm_topk_prob / routed_scaling_factor: MoE parameters.
        q_lora_rank / kv_lora_rank / qk_nope_head_dim / qk_rope_head_dim /
        v_head_dim: MLA dimensions.
        rope_interleave: Whether rotary dims are stored interleaved.
        llama4_beta / llama4_original_max: Attention-temperature parameters.
        norm_eps: Epsilon of all RMSNorms.

    Call args:
        hidden_states, cos, sin, positions, attention_mask, past_key_value,
        use_cache: as in :class:`Mistral4Attention`.

    Returns:
        The block output, or ``(output, (key, value))`` when ``use_cache``.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim_dense,
        moe_mlp_dim,
        num_heads,
        is_moe,
        num_experts,
        num_experts_per_tok,
        n_shared_experts,
        n_group,
        topk_group,
        norm_topk_prob,
        routed_scaling_factor,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rope_interleave=True,
        llama4_beta=0.1,
        llama4_original_max=8192,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim_dense = mlp_dim_dense
        self.moe_mlp_dim = moe_mlp_dim
        self.num_heads = num_heads
        self.is_moe = is_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_interleave = rope_interleave
        self.llama4_beta = llama4_beta
        self.llama4_original_max = llama4_original_max
        self.norm_eps = norm_eps

        self.attention_norm = Mistral4RMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Mistral4Attention(
            embed_dim,
            num_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            norm_eps,
            rope_interleave,
            llama4_beta,
            llama4_original_max,
            name="attention",
        )
        self.mlp_norm = Mistral4RMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = (
            Mistral4MoE(
                num_experts,
                num_experts_per_tok,
                embed_dim,
                moe_mlp_dim,
                n_shared_experts,
                n_group,
                topk_group,
                norm_topk_prob,
                routed_scaling_factor,
                name="mlp",
            )
            if is_moe
            else Mistral4MLP(embed_dim, mlp_dim_dense, name="mlp")
        )

    def call(
        self,
        hidden_states,
        cos,
        sin,
        positions,
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
            positions,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_kv) if use_cache else hidden_states

    def decode_step(
        self,
        hidden_states,
        cos,
        sin,
        positions,
        cache_k,
        cache_v,
        write_pos,
        key_mask,
    ):
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        attn_out, cache_k, cache_v = self.attention.decode_step(
            x, cos, sin, positions, cache_k, cache_v, write_pos, key_mask
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        x = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(x)
        return hidden_states, cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim_dense": self.mlp_dim_dense,
                "moe_mlp_dim": self.moe_mlp_dim,
                "num_heads": self.num_heads,
                "is_moe": self.is_moe,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "n_shared_experts": self.n_shared_experts,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "norm_topk_prob": self.norm_topk_prob,
                "routed_scaling_factor": self.routed_scaling_factor,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "rope_interleave": self.rope_interleave,
                "llama4_beta": self.llama4_beta,
                "llama4_original_max": self.llama4_original_max,
                "norm_eps": self.norm_eps,
            }
        )
        return config
