import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .config import GEMMA2_CONFIG, GEMMA2_WEIGHTS_URLS
from .gemma2_layers import Gemma2DecoderLayer, Gemma2RMSNorm

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma2Model(SubclassedBaseModel):
    """Gemma 2 decoder-only transformer backbone (no LM head).

    Gemma's scaled embeddings, ``(1 + w)`` RMSNorms, and GeGLU, plus the
    Gemma 2 additions: a four-norm sandwich around every residual branch,
    attention-logit tanh softcapping (50.0) applied before the mask,
    ``query_pre_attn_scalar`` attention scaling, and alternating
    sliding-window (even layers) / full (odd layers) causal attention.
    Returns raw features; use :class:`Gemma2Generate` for logits / text
    (which also applies the final-logit softcap, 30.0).

        model = Gemma2Model.from_weights("gemma-2-2b")
        out = model({"input_ids": ids})["last_hidden_state"]  # (B, L, embed_dim)

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: GeGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim.
        query_pre_attn_scalar: Attention scaling denominator.
        attn_logit_softcapping: Attention tanh softcap (``None`` disables).
        final_logit_softcapping: LM-head tanh softcap (``None`` disables).
        sliding_window: Window of the sliding (even) layers.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        tie_embeddings: Whether :class:`Gemma2Generate` ties the LM head
            (Gemma 2 checkpoints do).
    """

    HF_MODEL_TYPE = "gemma2"
    BASE_MODEL_CONFIG = GEMMA2_CONFIG
    BASE_WEIGHT_CONFIG = GEMMA2_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=256000,
        embed_dim=2304,
        mlp_dim=9216,
        num_layers=26,
        num_heads=8,
        num_kv_heads=4,
        head_dim=256,
        query_pre_attn_scalar=256.0,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        sliding_window=4096,
        norm_eps=1e-6,
        rope_theta=10000.0,
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
        self.head_dim = head_dim
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attn_logit_softcapping = attn_logit_softcapping
        self.final_logit_softcapping = final_logit_softcapping
        self.sliding_window = sliding_window
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Gemma2DecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                query_pre_attn_scalar,
                attn_logit_softcapping,
                norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Gemma2RMSNorm(eps=norm_eps, name="final_norm")

    def is_sliding(self, layer_idx):
        # HF: "sliding_attention" if bool((i + 1) % 2) -> even layers slide.
        return bool((layer_idx + 1) % 2)

    def embed_scaled(self, input_ids):
        return self.token_embedding(input_ids) * ops.cast(
            self.embed_dim**0.5, self.compute_dtype
        )

    def rope_tables(self, position_ids):
        hd = self.head_dim
        inv_freq = 1.0 / ops.power(
            self.rope_theta, ops.arange(0, hd, 2, dtype="float32") / hd
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
        cos, sin = self.rope_tables(position_ids)
        full_mask, sliding_mask = self.build_masks(seq, attention_mask)
        for i, layer in enumerate(self.decoder_layers):
            mask = sliding_mask if self.is_sliding(i) else full_mask
            hidden = layer(hidden, cos, sin, attention_mask=mask)
        return {"last_hidden_state": self.final_norm(hidden)}

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "mlp_dim": hf_config["intermediate_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_kv_heads": hf_config.get(
                "num_key_value_heads", hf_config["num_attention_heads"]
            ),
            "head_dim": hf_config.get("head_dim", 256),
            "query_pre_attn_scalar": hf_config.get("query_pre_attn_scalar", 256.0),
            "attn_logit_softcapping": hf_config.get("attn_logit_softcapping"),
            "final_logit_softcapping": hf_config.get("final_logit_softcapping"),
            "sliding_window": hf_config.get("sliding_window", 4096),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "tie_embeddings": hf_config.get("tie_word_embeddings", True),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma2_hf_to_keras import transfer_gemma2_weights

        transfer_gemma2_weights(keras_model, hf_state_dict)

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
                "head_dim": self.head_dim,
                "query_pre_attn_scalar": self.query_pre_attn_scalar,
                "attn_logit_softcapping": self.attn_logit_softcapping,
                "final_logit_softcapping": self.final_logit_softcapping,
                "sliding_window": self.sliding_window,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma2Generate(Gemma2Model, BaseGeneration):
    """Gemma 2 backbone + a (tied) LM head with final-logit softcapping and
    fast ``.generate()``.

    The vocabulary projection (tied token embedding) is followed by
    ``tanh(logits / 30) * 30`` when ``final_logit_softcapping`` is set —
    matching the Gemma 2 checkpoints. ``call`` returns both ``logits`` and
    ``last_hidden_state``. Fast generation comes from
    :class:`~kerasformers.base.BaseGeneration` via ``build_cache`` /
    ``call_with_cache``, respecting the per-layer full / sliding masks.
    Constructor ``Args`` are inherited from :class:`Gemma2Model`.

        gen = Gemma2Generate.from_weights("gemma-2-2b-it")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    # Gemma <eos> / <end_of_turn> stop ids. Explicit generate() args override.
    eos_token_id = (1, 107)

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
        # Parallel prefill into a fixed (B, num_layers, 2, num_kv_heads,
        # max_len, head_dim) cache with per-layer full / sliding masks.
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        if padding_mask is not None:
            am = ops.cast(padding_mask, "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos, sin = self.rope_tables(position_ids)
        full_mask, sliding_mask = self.build_masks(prompt_len, padding_mask)
        hidden = self.embed_scaled(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            mask = sliding_mask if self.is_sliding(i) else full_mask
            hidden, (k, v) = layer(
                hidden, cos, sin, attention_mask=mask, use_cache=True
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
        # One decode step; sliding layers see only (pos - window, pos].
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
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
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            km = sliding_km if self.is_sliding(i) else full_km
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[:, i, 0], cache[:, i, 1], pos, km
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
