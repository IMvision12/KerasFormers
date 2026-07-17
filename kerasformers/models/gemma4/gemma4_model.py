import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .gemma4_config import GEMMA4_CONFIG, GEMMA4_WEIGHTS_URLS
from .gemma4_layers import Gemma4DecoderLayer, Gemma4RMSNorm

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4Model(SubclassedBaseModel):
    """Gemma 4 text decoder backbone (no LM head).

    Gemma's scaled embeddings and ``(1 + w)`` norms with Gemma 4's per-layer
    attention geometry: sliding layers (5:1 pattern) use ``head_dim`` 256
    with full default rope (theta 1e4); global layers use
    ``global_head_dim`` 512 with few K/V heads, ``K = V`` attention (no value
    projection: the value is the weightlessly-normed key projection), and
    "proportional" *partial* rotary (the first quarter of the head, theta
    1e6). Attention scores are unscaled; per-head q/k norms carry the scale.
    Feed-forwards are GeGLU; on the 26B-A4B a parallel 128-expert top-8
    branch (per-expert-scaled router) is added. Each layer's output is
    multiplied by a learned ``layer_scalar``. The audio and vision towers of
    the omnimodal checkpoints are not ported: their ``model.*`` text weights
    load directly. E2B/E4B variants (per-layer inputs, shared KV) are not
    supported. Returns raw features; use :class:`Gemma4Generate`.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: Dense GeGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: K/V heads on sliding layers.
        num_global_kv_heads: K/V heads on global layers.
        head_dim: Sliding-layer per-head dim (256).
        global_head_dim: Global-layer per-head dim (512).
        k_eq_v: Global layers reuse the key projection as the value.
        enable_moe: Whether layers carry the parallel expert branch.
        num_experts / num_experts_per_tok / moe_mlp_dim: MoE parameters.
        sliding_window: Window of the sliding layers.
        sliding_window_pattern: Every ``pattern``-th layer is global (6).
        partial_rotary_factor: Fraction of the global head that is rotated.
        final_logit_softcapping: LM-head tanh softcap (30.0).
        norm_eps: RMSNorm epsilon.
        rope_theta: Global-layer rotary base (1e6).
        rope_local_theta: Sliding-layer rotary base (1e4).
        tie_embeddings: Whether :class:`Gemma4Generate` ties the LM head.
    """

    HF_MODEL_TYPE = ("gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text")
    BASE_MODEL_CONFIG = GEMMA4_CONFIG
    BASE_WEIGHT_CONFIG = GEMMA4_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=262144,
        embed_dim=3840,
        mlp_dim=15360,
        num_layers=48,
        num_heads=16,
        num_kv_heads=8,
        num_global_kv_heads=1,
        head_dim=256,
        global_head_dim=512,
        k_eq_v=True,
        enable_moe=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_mlp_dim=0,
        sliding_window=1024,
        sliding_window_pattern=6,
        partial_rotary_factor=0.25,
        final_logit_softcapping=30.0,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_local_theta=10000.0,
        tie_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_global_kv_heads = num_global_kv_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.k_eq_v = k_eq_v
        self.enable_moe = enable_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.partial_rotary_factor = partial_rotary_factor
        self.final_logit_softcapping = final_logit_softcapping
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.rope_local_theta = rope_local_theta
        self.tie_embeddings = tie_embeddings
        self.global_rot_dim = 2 * int(partial_rotary_factor * global_head_dim // 2)

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = []
        for i in range(num_layers):
            sliding = self.is_sliding(i)
            self.decoder_layers.append(
                Gemma4DecoderLayer(
                    embed_dim,
                    mlp_dim,
                    num_heads,
                    num_kv_heads if sliding else num_global_kv_heads,
                    head_dim if sliding else global_head_dim,
                    k_eq_v=(not sliding) and k_eq_v,
                    is_moe=enable_moe,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    moe_mlp_dim=moe_mlp_dim,
                    norm_eps=norm_eps,
                    name=f"decoder_layer_{i}",
                )
            )
        self.final_norm = Gemma4RMSNorm(eps=norm_eps, name="final_norm")

    def is_sliding(self, layer_idx):
        return bool((layer_idx + 1) % self.sliding_window_pattern)

    def embed_scaled(self, input_ids):
        return self.token_embedding(input_ids) * ops.cast(
            self.embed_dim**0.5, self.compute_dtype
        )

    def rope_tables(self, position_ids, local):
        # Sliding layers: full-width default rope over head_dim, theta 1e4.
        # Global layers: "proportional" partial rope, frequencies for the
        # first ``global_rot_dim`` dims (exponent / head_dim), zero-padded to
        # head_dim // 2 so the padded dims pass through unrotated (HF scheme:
        # cos(0) = 1, sin(0) = 0).
        if local:
            hd, rot = self.head_dim, self.head_dim
            theta = self.rope_local_theta
        else:
            hd, rot = self.global_head_dim, self.global_rot_dim
            theta = self.rope_theta
        inv_freq = 1.0 / ops.power(theta, ops.arange(0, rot, 2, dtype="float32") / hd)
        if rot < hd:
            inv_freq = ops.concatenate(
                [inv_freq, ops.zeros(((hd - rot) // 2,), dtype="float32")], axis=0
            )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        return (
            ops.cast(ops.cos(emb), self.compute_dtype),
            ops.cast(ops.sin(emb), self.compute_dtype),
        )

    def build_masks(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = ki <= qi
        full = ops.cast(ops.where(causal, 0.0, MASK_NEG), "float32")[None, None]
        sliding_keep = ops.logical_and(causal, ki > qi - self.sliding_window)
        sliding = ops.cast(ops.where(sliding_keep, 0.0, MASK_NEG), "float32")[
            None, None
        ]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            pad = (1.0 - am)[:, None, None, :] * MASK_NEG
            full = full + pad
            sliding = sliding + pad
        return full, sliding

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        attention_mask = inputs.get("attention_mask")
        hidden = self.embed_scaled(input_ids)
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos_l, sin_l = self.rope_tables(position_ids, local=True)
        cos_g, sin_g = self.rope_tables(position_ids, local=False)
        full_mask, sliding_mask = self.build_masks(seq, attention_mask)
        for i, layer in enumerate(self.decoder_layers):
            if self.is_sliding(i):
                hidden = layer(hidden, cos_l, sin_l, attention_mask=sliding_mask)
            else:
                hidden = layer(hidden, cos_g, sin_g, attention_mask=full_mask)
        return {"last_hidden_state": self.final_norm(hidden)}

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config.get("text_config", hf_config)
        if text.get("hidden_size_per_layer_input") or text.get("num_kv_shared_layers"):
            raise NotImplementedError(
                "Gemma 4 E-variants (per-layer inputs / shared KV layers) are "
                "not supported by this port."
            )
        rope = text.get("rope_parameters") or {}
        full_rope = rope.get("full_attention") or {}
        sliding_rope = rope.get("sliding_attention") or {}
        return {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text.get(
                "num_key_value_heads", text["num_attention_heads"]
            ),
            "num_global_kv_heads": text.get("num_global_key_value_heads")
            or text.get("num_key_value_heads", 1),
            "head_dim": text.get("head_dim", 256),
            "global_head_dim": text.get("global_head_dim", 512),
            "k_eq_v": bool(text.get("attention_k_eq_v", False)),
            "enable_moe": bool(text.get("enable_moe_block", False)),
            "num_experts": text.get("num_experts") or 0,
            "num_experts_per_tok": text.get("top_k_experts") or 0,
            "moe_mlp_dim": text.get("moe_intermediate_size") or 0,
            "sliding_window": text.get("sliding_window", 1024),
            "sliding_window_pattern": text.get("sliding_window_pattern", 6),
            "partial_rotary_factor": full_rope.get("partial_rotary_factor", 0.25),
            "final_logit_softcapping": text.get("final_logit_softcapping"),
            "norm_eps": text.get("rms_norm_eps", 1e-6),
            "rope_theta": full_rope.get(
                "rope_theta", text.get("rope_theta", 1000000.0)
            ),
            "rope_local_theta": sliding_rope.get(
                "rope_theta", text.get("rope_local_base_freq", 10000.0)
            ),
            "tie_embeddings": text.get("tie_word_embeddings", True),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma4_hf_to_keras import transfer_gemma4_weights

        transfer_gemma4_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "num_global_kv_heads": self.num_global_kv_heads,
                "head_dim": self.head_dim,
                "global_head_dim": self.global_head_dim,
                "k_eq_v": self.k_eq_v,
                "enable_moe": self.enable_moe,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "sliding_window": self.sliding_window,
                "sliding_window_pattern": self.sliding_window_pattern,
                "partial_rotary_factor": self.partial_rotary_factor,
                "final_logit_softcapping": self.final_logit_softcapping,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "rope_local_theta": self.rope_local_theta,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4Generate(Gemma4Model, BaseGeneration):
    """Gemma 4 text backbone + a (tied) LM head with final-logit softcapping
    and fast ``.generate()``.

    The vocabulary projection (tied token embedding) is followed by
    ``tanh(logits / 30) * 30``. Fast generation comes from
    :class:`~kerasformers.base.BaseGeneration` via ``build_cache`` /
    ``call_with_cache`` over per-layer-geometry caches (sliding layers store
    ``head_dim``-wide K/V, global layers ``global_head_dim``-wide), kept as a
    per-layer tuple. Constructor ``Args`` are inherited from
    :class:`Gemma4Model`.
    """

    # Gemma <eos> / <end_of_turn> stop ids. Explicit generate() args override.
    eos_token_id = (1, 106)

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
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = ops.tanh(logits / cap) * cap
        return logits

    def call(self, inputs):
        hidden = super().call(inputs)["last_hidden_state"]
        return {"logits": self.project(hidden), "last_hidden_state": hidden}

    def build_cache(self, token_ids, padding_mask, max_len):
        # Parallel prefill; per-layer caches (tuple) because sliding and
        # global layers have different K/V geometry.
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        if padding_mask is not None:
            am = ops.cast(padding_mask, "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos_l, sin_l = self.rope_tables(position_ids, local=True)
        cos_g, sin_g = self.rope_tables(position_ids, local=False)
        full_mask, sliding_mask = self.build_masks(prompt_len, padding_mask)
        hidden = self.embed_scaled(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            if self.is_sliding(i):
                hidden, (k, v) = layer(
                    hidden, cos_l, sin_l, attention_mask=sliding_mask, use_cache=True
                )
            else:
                hidden, (k, v) = layer(
                    hidden, cos_g, sin_g, attention_mask=full_mask, use_cache=True
                )
            nkv = int(k.shape[1])
            hd = int(k.shape[3])
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return tuple(layer_caches), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        # One decode step over the per-layer cache tuple.
        batch = int(token_ids.shape[0])
        max_len = int(cache[0].shape[3])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos_l, sin_l = self.rope_tables(positions, local=True)
        cos_g, sin_g = self.rope_tables(positions, local=False)
        ar = ops.arange(max_len)
        full_km = ops.cast(ops.where(ar <= pos, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        sliding_km = ops.cast(
            ops.where(
                ops.logical_and(ar <= pos, ar > pos - self.sliding_window),
                0.0,
                MASK_NEG,
            ),
            "float32",
        )[None, None, None, :]
        h = self.embed_scaled(token_ids)
        new_caches = []
        for i, layer in enumerate(self.decoder_layers):
            if self.is_sliding(i):
                h, ck, cv = layer.decode_step(
                    h, cos_l, sin_l, cache[i][:, 0], cache[i][:, 1], pos, sliding_km
                )
            else:
                h, ck, cv = layer.decode_step(
                    h, cos_g, sin_g, cache[i][:, 0], cache[i][:, 1], pos, full_km
                )
            new_caches.append(ops.stack([ck, cv], axis=1))
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, tuple(new_caches)
