import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .cohere2_moe_layers import Cohere2MoeDecoderLayer, make_norm
from .config import COHERE2_MOE_CONFIG, COHERE2_MOE_WEIGHTS_URLS

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2MoeModel(SubclassedBaseModel):
    """Cohere2-MoE sparse decoder backbone (no LM head).

    Cohere2 token mixer (parallel attn+MLP, NoPE full layers + sliding rope
    layers, ``logit_scale``) with a Cohere MoE block on the sparse layers: a
    **top-k-first** router (softmax over the k, or sigmoid + optional norm)
    plus optional shared expert(s) combined by sum/average. The first
    ``first_k_dense_replace`` layers are dense (and force rope). Norm is
    RMSNorm when ``rms_norm_eps`` is set, else Cohere LayerNorm. Returns raw
    features; use :class:`Cohere2MoeGenerate` for logits.

    Args:
        vocab_size / embed_dim / num_layers / num_heads / num_kv_heads /
        head_dim / mlp_dim: Geometry (``mlp_dim`` is the dense-layer hidden width).
        num_experts / num_experts_per_tok: Routed-expert count and top-k.
        moe_mlp_dim: Per-expert hidden width; defaults to ``mlp_dim``.
        expert_selection_fn: ``"softmax"`` (over the top-k) or ``"sigmoid"``.
        norm_topk_prob: Renormalize the selected weights (sigmoid only).
        num_shared_experts: Always-on shared experts (0 disables).
        shared_combine: ``"sum"`` or ``"average"`` for the shared-expert merge.
        first_k_dense_replace: Number of leading dense (non-MoE) layers.
        prefix_dense_intermediate_size: Hidden width of those dense layers;
            defaults to ``mlp_dim``.
        prefix_dense_sliding_window_pattern: Attention pattern within the dense
            prefix; ``1`` makes every prefix layer sliding + force-rope.
        sliding_window: Window size for the sliding layers.
        sliding_window_pattern: Every Nth non-prefix layer is full attention (NoPE).
        rms_norm_eps: When set, use RMSNorm with this epsilon; otherwise use
            mean-centered Cohere LayerNorm with ``norm_eps``.
        norm_eps: LayerNorm epsilon (used when ``rms_norm_eps`` is ``None``).
        rope_theta: Rotary base frequency.
        attention_bias: Attention projection bias.
        logit_scale: Output-logit multiplier.
        tie_embeddings: Whether the head ties to the token embedding.
    """

    HF_MODEL_TYPE = "cohere2_moe"
    BASE_MODEL_CONFIG = COHERE2_MOE_CONFIG
    BASE_WEIGHT_CONFIG = COHERE2_MOE_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=256000,
        embed_dim=8192,
        num_layers=40,
        num_heads=64,
        num_kv_heads=64,
        head_dim=128,
        mlp_dim=22528,
        num_experts=8,
        num_experts_per_tok=2,
        moe_mlp_dim=None,
        expert_selection_fn="softmax",
        norm_topk_prob=True,
        num_shared_experts=0,
        shared_combine="average",
        first_k_dense_replace=0,
        prefix_dense_intermediate_size=None,
        prefix_dense_sliding_window_pattern=1,
        sliding_window=4096,
        sliding_window_pattern=4,
        rms_norm_eps=None,
        norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        logit_scale=0.0625,
        tie_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.mlp_dim = mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim or mlp_dim
        self.expert_selection_fn = expert_selection_fn
        self.norm_topk_prob = norm_topk_prob
        self.num_shared_experts = num_shared_experts
        self.shared_combine = shared_combine
        self.first_k_dense_replace = first_k_dense_replace
        self.prefix_dense_intermediate_size = prefix_dense_intermediate_size or mlp_dim
        self.prefix_dense_sliding_window_pattern = prefix_dense_sliding_window_pattern
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.rms_norm_eps = rms_norm_eps
        self.norm_eps = norm_eps
        self.use_rms_norm = rms_norm_eps is not None
        self.eps = rms_norm_eps if self.use_rms_norm else norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.logit_scale = logit_scale
        self.tie_embeddings = tie_embeddings

        prefix = [
            "sliding_attention"
            if ((i + 1) % prefix_dense_sliding_window_pattern) != 0
            else "full_attention"
            for i in range(first_k_dense_replace)
        ]
        rest = [
            "sliding_attention"
            if ((i + 1) % sliding_window_pattern) != 0
            else "full_attention"
            for i in range(num_layers - first_k_dense_replace)
        ]
        self.layer_types = tuple(prefix + rest)
        self.mlp_layer_types = tuple(
            "dense" if i < first_k_dense_replace else "sparse"
            for i in range(num_layers)
        )

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [self.make_layer(i) for i in range(num_layers)]
        self.final_norm = make_norm(self.use_rms_norm, self.eps, "final_norm")

    def make_layer(self, i):
        is_dense = self.mlp_layer_types[i] == "dense"
        layer_type = self.layer_types[i]
        force_rope = is_dense and self.prefix_dense_sliding_window_pattern == 1
        use_rope = layer_type == "sliding_attention" or force_rope
        return Cohere2MoeDecoderLayer(
            self.embed_dim,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            layer_type,
            use_rope,
            use_moe=not is_dense,
            dense_mlp_dim=self.prefix_dense_intermediate_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_mlp_dim=self.moe_mlp_dim,
            expert_selection_fn=self.expert_selection_fn,
            norm_topk_prob=self.norm_topk_prob,
            num_shared_experts=self.num_shared_experts,
            shared_mlp_dim=self.mlp_dim * max(self.num_shared_experts, 1),
            shared_combine=self.shared_combine,
            use_rms_norm=self.use_rms_norm,
            norm_eps=self.eps,
            attention_bias=self.attention_bias,
            name=f"decoder_layer_{i}",
        )

    def rope_tables(self, position_ids):
        hd = self.head_dim
        inv_freq = 1.0 / ops.power(
            self.rope_theta, ops.arange(0, hd, 2, dtype="float32") / hd
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.repeat(freqs, 2, axis=-1)
        return (
            ops.cast(ops.cos(emb), self.compute_dtype),
            ops.cast(ops.sin(emb), self.compute_dtype),
        )

    def build_masks(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = ops.where(ki <= qi, 0.0, MASK_NEG)
        sliding = ops.where(
            ops.logical_and(ki <= qi, (qi - ki) < self.sliding_window), 0.0, MASK_NEG
        )
        full = ops.cast(causal, "float32")[None, None]
        slide = ops.cast(sliding, "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            pad = (1.0 - am)[:, None, None, :] * MASK_NEG
            full = full + pad
            slide = slide + pad
        return {"full_attention": full, "sliding_attention": slide}

    def forward_features(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        hidden = self.token_embedding(input_ids)
        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables(position_ids)
        masks = self.build_masks(seq, inputs.get("attention_mask"))
        for layer in self.decoder_layers:
            hidden = layer(hidden, cos, sin, attention_mask=masks[layer.layer_type])
        return self.final_norm(hidden)

    def call(self, inputs):
        return {"last_hidden_state": self.forward_features(inputs)}

    @classmethod
    def config_from_hf(cls, hf_config):
        rope = hf_config.get("rope_parameters") or {}
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_kv_heads": hf_config.get(
                "num_key_value_heads", hf_config["num_attention_heads"]
            ),
            "head_dim": hf_config.get("head_dim", 128),
            "mlp_dim": hf_config["intermediate_size"],
            "num_experts": hf_config.get("num_experts", 8),
            "num_experts_per_tok": hf_config.get("num_experts_per_tok", 2),
            "moe_mlp_dim": hf_config.get("intermediate_size"),
            "expert_selection_fn": hf_config.get("expert_selection_fn", "softmax"),
            "norm_topk_prob": bool(hf_config.get("norm_topk_prob", True)),
            "num_shared_experts": hf_config.get("num_shared_experts", 0),
            "shared_combine": hf_config.get(
                "shared_expert_combination_strategy", "average"
            ),
            "first_k_dense_replace": hf_config.get("first_k_dense_replace", 0),
            "prefix_dense_intermediate_size": hf_config.get(
                "prefix_dense_intermediate_size"
            ),
            "prefix_dense_sliding_window_pattern": hf_config.get(
                "prefix_dense_sliding_window_pattern", 1
            ),
            "sliding_window": hf_config.get("sliding_window", 4096),
            "sliding_window_pattern": hf_config.get("sliding_window_pattern", 4),
            "rms_norm_eps": hf_config.get("rms_norm_eps"),
            "norm_eps": hf_config.get("layer_norm_eps", 1e-5),
            "rope_theta": rope.get("rope_theta", hf_config.get("rope_theta", 10000.0)),
            "attention_bias": bool(hf_config.get("attention_bias") or False),
            "logit_scale": hf_config.get("logit_scale", 0.0625),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings", True)),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_cohere2_moe_hf_to_keras import transfer_cohere2_moe_weights

        transfer_cohere2_moe_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "mlp_dim": self.mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "expert_selection_fn": self.expert_selection_fn,
                "norm_topk_prob": self.norm_topk_prob,
                "num_shared_experts": self.num_shared_experts,
                "shared_combine": self.shared_combine,
                "first_k_dense_replace": self.first_k_dense_replace,
                "prefix_dense_intermediate_size": self.prefix_dense_intermediate_size,
                "prefix_dense_sliding_window_pattern": (
                    self.prefix_dense_sliding_window_pattern
                ),
                "sliding_window": self.sliding_window,
                "sliding_window_pattern": self.sliding_window_pattern,
                "rms_norm_eps": self.rms_norm_eps,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "attention_bias": self.attention_bias,
                "logit_scale": self.logit_scale,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2MoeGenerate(Cohere2MoeModel, BaseGeneration):
    """Cohere2-MoE (North-Mini / Command-MoE) with a language-model head + fast ``.generate()``.

    Adds a vocabulary projection on top of :class:`Cohere2MoeModel`: a bias-free
    ``lm_head`` when ``tie_embeddings`` is ``False``, otherwise the tied token
    embedding; either way logits are scaled by ``logit_scale``. ``call`` returns
    both ``logits`` and the final ``last_hidden_state``.

    Fast generation uses :class:`~kerasformers.base.BaseGeneration`'s fixed-cache
    compiled loop: :meth:`build_cache` prefills the prompt into a full-length
    per-layer KV cache, then :meth:`call_with_cache` decodes one token at a time.
    The cache is full-length for **every** layer; the sliding-window layers
    enforce their window through the decode key-mask (keys older than
    ``sliding_window`` are masked) and the full/NoPE layers see all keys, so the
    loop stays constant-shape. The dense-vs-MoE feed-forward split is unchanged
    from :class:`Cohere2MoeModel`. ``eos_token_id`` defaults to Cohere's
    ``<|END_OF_TURN_TOKEN|>`` (255001); pass ``eos_token_id`` to :meth:`generate`
    to override.

    Construction mirrors :class:`Cohere2MoeModel`::

        gen = Cohere2MoeGenerate.from_weights("hf:CohereLabs/North-Mini-Code-1.0")
        out = gen.generate(input_ids, max_new_tokens=64)
    """

    eos_token_id = (255001,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = (
            None
            if self.tie_embeddings
            else layers.Dense(self.vocab_size, use_bias=False, name="lm_head")
        )

    def project(self, hidden):
        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))
        return logits * self.logit_scale

    def call(self, inputs):
        hidden = self.forward_features(inputs)
        return {"logits": self.project(hidden), "last_hidden_state": hidden}

    def build_cache(self, token_ids, padding_mask, max_len):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos_p, sin_p = self.rope_tables(position_ids)
        masks = self.build_masks(prompt_len, padding_mask)
        hidden = self.token_embedding(ops.cast(token_ids, "int32"))
        layer_caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden,
                cos_p,
                sin_p,
                attention_mask=masks[layer.layer_type],
                use_cache=True,
            )
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return cache, logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos_t, sin_t = self.rope_tables(positions)
        ar = ops.arange(max_len)
        full_mask = ops.cast(ops.where(ar <= pos, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        slide_mask = ops.cast(
            ops.where(
                ops.logical_and(ar <= pos, (pos - ar) < self.sliding_window),
                0.0,
                MASK_NEG,
            ),
            "float32",
        )[None, None, None, :]
        masks = {"full_attention": full_mask, "sliding_attention": slide_mask}
        h = self.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h,
                cos_t,
                sin_t,
                cache[:, i, 0],
                cache[:, i, 1],
                pos,
                masks[layer.layer_type],
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
