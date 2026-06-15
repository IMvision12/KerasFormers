import math

import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .config import LLAMA4_CONFIG, LLAMA4_WEIGHTS_URLS
from .llama4_layers import Llama4DecoderLayer, Llama4RMSNorm

MASK_NEG = -1e9


def llama4_scaled_inv_freq(
    head_dim, base, factor, low_freq_factor, high_freq_factor, original_max_pos
):
    """Llama-3-style frequency-banded rope scaling used by the Scout model.

    Long-wavelength bands are divided by ``factor``, short-wavelength bands
    are kept, and the band in between is smoothly interpolated (Scout sets
    ``low_freq_factor == high_freq_factor == 1.0``, collapsing the
    interpolated band to nothing). Returns the ``(head_dim // 2,)``
    inverse-frequency tensor.
    """
    inv_freq = 1.0 / ops.power(
        float(base), ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    low_freq_wavelen = original_max_pos / low_freq_factor
    high_freq_wavelen = original_max_pos / high_freq_factor
    wavelen = 2.0 * math.pi / inv_freq
    scaled = ops.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    denom = high_freq_factor - low_freq_factor
    denom = denom if denom != 0 else 1.0
    smooth = (original_max_pos / wavelen - low_freq_factor) / denom
    smoothed = (1.0 - smooth) * scaled / factor + smooth * scaled
    is_medium = ops.logical_and(
        wavelen >= high_freq_wavelen, wavelen <= low_freq_wavelen
    )
    return ops.where(is_medium, smoothed, scaled)


@keras.saving.register_keras_serializable(package="kerasformers")
class Llama4Model(SubclassedBaseModel):
    """Llama 4 text decoder backbone (the ``language_model`` of Scout / Maverick).

    ``token_embedding -> num_layers x Llama4DecoderLayer -> final RMSNorm``
    with Llama 4's iRoPE attention scheme: every
    ``no_rope_layer_interval``-th layer (the 4th, 8th, ...) is a NoPE layer —
    no rotary, full causal attention, and position-dependent attention
    temperature scaling — while the remaining layers apply the
    interleaved-pair rotary embedding and attend within
    ``attention_chunk_size``-token chunks (chunked causal). The feed-forward
    is a sigmoid-top-1-routed mixture of experts with an always-active shared
    expert on every layer for Scout (``interleave_moe_layer_step=1``,
    16 experts) and on alternating layers for Maverick (step 2, 128 experts;
    the rest are dense ``intermediate_size_mlp`` SwiGLUs). Scout additionally
    L2-normalizes rotated q/k (``use_qk_norm``) and scales its rope inverse
    frequencies with the llama3 banded scheme. This port evaluates all experts
    densely and combines by the routing scores — mathematically identical to
    sparse routing, compute O(num_experts). Text-only: the vision tower of the
    multimodal checkpoints is not ported; their ``language_model.*`` weights
    load directly. Returns raw features; use :class:`Llama4Generate` for
    logits / text.

        model = Llama4Model.from_weights("llama4-scout-17b-16e")
        out = model({"input_ids": ids})["last_hidden_state"]  # (B, L, embed_dim)

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: Per-expert and shared-expert hidden width.
        dense_mlp_dim: Dense (non-MoE) layers' feed-forward hidden width.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.
        num_experts: Routed expert count (Scout 16, Maverick 128).
        num_experts_per_tok: Top-k experts per token (released models: 1).
        interleave_moe_layer_step: Every ``step``-th layer is MoE (1 = all).
        no_rope_layer_interval: Every ``interval``-th layer is NoPE.
        attention_chunk_size: Chunk width of the rope layers' local attention.
        use_qk_norm: L2-normalize rotated q/k on rope layers (Scout).
        attn_temperature_tuning: Scale NoPE-layer queries by the
            position-dependent temperature.
        floor_scale, attn_scale: Temperature-tuning constants.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        rope_factor: llama3 rope-scaling factor; ``None`` disables scaling
            (Maverick). 16.0 for Scout.
        rope_low_freq_factor / rope_high_freq_factor: scaling band edges.
        rope_original_max_pos: Pretraining context the bands are relative to.
        tie_embeddings: Whether :class:`Llama4Generate` ties the LM head
            (released checkpoints: ``False``).
    """

    HF_MODEL_TYPE = ("llama4", "llama4_text")
    BASE_MODEL_CONFIG = LLAMA4_CONFIG
    BASE_WEIGHT_CONFIG = LLAMA4_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=202048,
        embed_dim=5120,
        mlp_dim=8192,
        dense_mlp_dim=16384,
        num_layers=48,
        num_heads=40,
        num_kv_heads=8,
        head_dim=128,
        num_experts=16,
        num_experts_per_tok=1,
        interleave_moe_layer_step=1,
        no_rope_layer_interval=4,
        attention_chunk_size=8192,
        use_qk_norm=True,
        attn_temperature_tuning=True,
        floor_scale=8192.0,
        attn_scale=0.1,
        norm_eps=1e-5,
        rope_theta=500000.0,
        rope_factor=16.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=1.0,
        rope_original_max_pos=8192,
        tie_embeddings=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.dense_mlp_dim = dense_mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.interleave_moe_layer_step = interleave_moe_layer_step
        self.no_rope_layer_interval = no_rope_layer_interval
        self.attention_chunk_size = attention_chunk_size
        self.use_qk_norm = use_qk_norm
        self.attn_temperature_tuning = attn_temperature_tuning
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor
        self.rope_low_freq_factor = rope_low_freq_factor
        self.rope_high_freq_factor = rope_high_freq_factor
        self.rope_original_max_pos = rope_original_max_pos
        self.tie_embeddings = tie_embeddings

        moe_layers = set(
            range(interleave_moe_layer_step - 1, num_layers, interleave_moe_layer_step)
        )
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Llama4DecoderLayer(
                embed_dim,
                mlp_dim,
                dense_mlp_dim,
                num_heads,
                num_kv_heads,
                self.head_dim,
                is_moe=i in moe_layers,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                use_rope=self.layer_uses_rope(i),
                use_qk_norm=use_qk_norm,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Llama4RMSNorm(eps=norm_eps, name="final_norm")

    def layer_uses_rope(self, layer_idx):
        # HF: no_rope_layers[i] = int((i + 1) % interval != 0); truthy = rope.
        return (layer_idx + 1) % self.no_rope_layer_interval != 0

    def rope_tables(self, position_ids):
        # Interleaved-pair cos/sin tables, (batch, len, head_dim // 2), in the
        # compute dtype, with llama3-banded inverse frequencies when
        # rope_factor is set (Scout) and plain ones otherwise (Maverick).
        hd = self.head_dim
        if self.rope_factor is not None:
            inv_freq = llama4_scaled_inv_freq(
                hd,
                self.rope_theta,
                self.rope_factor,
                self.rope_low_freq_factor,
                self.rope_high_freq_factor,
                self.rope_original_max_pos,
            )
        else:
            inv_freq = 1.0 / ops.power(
                self.rope_theta, ops.arange(0, hd, 2, dtype="float32") / hd
            )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        return (
            ops.cast(ops.cos(freqs), self.compute_dtype),
            ops.cast(ops.sin(freqs), self.compute_dtype),
        )

    def temperature_scales(self, positions):
        # Position-dependent attention-temperature scaling for NoPE layers
        # (HF keys these on the raw cache position, not the padding-aware
        # position ids): log1p(floor((pos + 1) / floor_scale)) * scale + 1.
        if not self.attn_temperature_tuning:
            return None
        pos = ops.cast(positions, "float32")
        return (
            ops.log(1.0 + ops.floor((pos + 1.0) / self.floor_scale)) * self.attn_scale
            + 1.0
        )

    def build_masks(self, seq, padding_mask):
        # (full, chunked) additive masks, each (1 or B, 1, seq, seq): full
        # causal for the NoPE layers, chunk-local causal for the rope layers.
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = ki <= qi
        full = ops.cast(ops.where(causal, 0.0, MASK_NEG), "float32")[None, None]
        same_chunk = (qi // self.attention_chunk_size) == (
            ki // self.attention_chunk_size
        )
        chunked = ops.cast(
            ops.where(ops.logical_and(causal, same_chunk), 0.0, MASK_NEG), "float32"
        )[None, None]
        if padding_mask is not None:
            pad = (1.0 - ops.cast(padding_mask, "float32"))[:, None, None, :] * MASK_NEG
            full = full + pad
            chunked = chunked + pad
        return full, chunked

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        attention_mask = inputs.get("attention_mask")
        am = (
            None
            if attention_mask is None
            else ops.cast(ops.convert_to_tensor(attention_mask), "int32")
        )
        hidden = self.token_embedding(input_ids)
        if am is not None:
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables(position_ids)
        scales = self.temperature_scales(ops.arange(seq))
        attn_scales = None if scales is None else scales[None, :, None, None]
        full_mask, chunked_mask = self.build_masks(seq, am)
        for i, layer in enumerate(self.decoder_layers):
            mask = chunked_mask if self.layer_uses_rope(i) else full_mask
            hidden = layer(
                hidden, cos, sin, attn_scales=attn_scales, attention_mask=mask
            )
        return {"last_hidden_state": self.final_norm(hidden)}

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config.get("text_config", hf_config)
        rope_scaling = text.get("rope_scaling") or {}
        return {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "dense_mlp_dim": text["intermediate_size_mlp"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text["num_key_value_heads"],
            "head_dim": text.get("head_dim"),
            "num_experts": text["num_local_experts"],
            "num_experts_per_tok": text.get("num_experts_per_tok", 1),
            "interleave_moe_layer_step": text.get("interleave_moe_layer_step", 1),
            "no_rope_layer_interval": text.get("no_rope_layer_interval", 4),
            "attention_chunk_size": text.get("attention_chunk_size", 8192),
            "use_qk_norm": text.get("use_qk_norm", True),
            "attn_temperature_tuning": bool(text.get("attn_temperature_tuning", True)),
            "floor_scale": text.get("floor_scale", 8192.0),
            "attn_scale": text.get("attn_scale", 0.1),
            "norm_eps": text.get("rms_norm_eps", 1e-5),
            "rope_theta": text.get("rope_theta", 500000.0),
            "rope_factor": rope_scaling.get("factor"),
            "rope_low_freq_factor": rope_scaling.get("low_freq_factor", 1.0),
            "rope_high_freq_factor": rope_scaling.get("high_freq_factor", 1.0),
            "rope_original_max_pos": rope_scaling.get(
                "original_max_position_embeddings", 8192
            ),
            "tie_embeddings": bool(text.get("tie_word_embeddings") or False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_llama4_hf_to_keras import transfer_llama4_weights

        transfer_llama4_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "dense_mlp_dim": self.dense_mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "interleave_moe_layer_step": self.interleave_moe_layer_step,
                "no_rope_layer_interval": self.no_rope_layer_interval,
                "attention_chunk_size": self.attention_chunk_size,
                "use_qk_norm": self.use_qk_norm,
                "attn_temperature_tuning": self.attn_temperature_tuning,
                "floor_scale": self.floor_scale,
                "attn_scale": self.attn_scale,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "rope_factor": self.rope_factor,
                "rope_low_freq_factor": self.rope_low_freq_factor,
                "rope_high_freq_factor": self.rope_high_freq_factor,
                "rope_original_max_pos": self.rope_original_max_pos,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Llama4Generate(Llama4Model, BaseGeneration):
    """Llama 4 text backbone + a language-model head and fast ``.generate()``.

    Adds a bias-free ``lm_head`` on top of :class:`Llama4Model` (the released
    checkpoints do not tie embeddings). ``call`` returns both ``logits``
    ``(batch, seq, vocab_size)`` and ``last_hidden_state``. Fast generation
    comes from :class:`~kerasformers.base.BaseGeneration`, fulfilled here by
    ``build_cache`` (parallel prefill into a fixed KV cache) and
    ``call_with_cache`` (one compiled decode step) — both respect the
    per-layer full / chunked masks, the NoPE layers' temperature scaling, and
    the rope layers' interleaved rotary. Constructor ``Args`` are inherited
    from :class:`Llama4Model`.

        gen = Llama4Generate.from_weights("llama4-scout-17b-16e-instruct")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    # Llama 4 stop ids: <|end_of_text|>, <|eom|>, <|eot|>. Explicit generate()
    # args override this.
    eos_token_id = (200001, 200007, 200008)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = (
            None
            if self.tie_embeddings
            else layers.Dense(self.vocab_size, use_bias=False, name="lm_head")
        )

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def call(self, inputs):
        hidden = super().call(inputs)["last_hidden_state"]
        return {"logits": self.project(hidden), "last_hidden_state": hidden}

    def build_cache(self, token_ids, padding_mask, max_len):
        # Parallel prefill: run the prompt and write each layer's K/V into a
        # pre-allocated (B, num_layers, 2, num_kv_heads, max_len, head_dim)
        # cache, with the per-layer full / chunked causal mask. Returns
        # (cache, last-token logits).
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        am = None if padding_mask is None else ops.cast(padding_mask, "int32")
        if am is not None:
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos, sin = self.rope_tables(position_ids)
        scales = self.temperature_scales(ops.arange(prompt_len))
        attn_scales = None if scales is None else scales[None, :, None, None]
        full_mask, chunked_mask = self.build_masks(prompt_len, am)
        hidden = self.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            mask = chunked_mask if self.layer_uses_rope(i) else full_mask
            hidden, (k, v) = layer(
                hidden,
                cos,
                sin,
                attn_scales=attn_scales,
                attention_mask=mask,
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
        # One decode step: NoPE layers see slots [0, pos] (plus temperature
        # scaling at pos); rope layers see only the slots in pos's chunk.
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        scales = self.temperature_scales(ops.reshape(pos, (1,)))
        attn_scales = None if scales is None else scales[None, :, None, None]
        ar = ops.arange(max_len)
        full_km = ops.cast(ops.where(ar <= pos, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        same_chunk = (ar // self.attention_chunk_size) == (
            pos // self.attention_chunk_size
        )
        chunked_km = ops.cast(
            ops.where(ops.logical_and(ar <= pos, same_chunk), 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            km = chunked_km if self.layer_uses_rope(i) else full_km
            h, ck, cv = layer.decode_step(
                h, cos, sin, attn_scales, cache[:, i, 0], cache[:, i, 1], pos, km
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
