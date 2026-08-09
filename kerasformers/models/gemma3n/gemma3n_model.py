import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .gemma3n_config import Gemma3nConfig, Gemma3nTextConfig
from .gemma3n_layers import (
    MNV5_ARCH,
    ConvNormAct,
    Gemma3nAudioConformerBlock,
    Gemma3nAudioSubSampleConvProjection,
    Gemma3nDecoderLayer,
    Gemma3nMultimodalEmbedder,
    Gemma3nRMSNorm,
    MobileNetV5MSFA,
    build_block,
)

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma3nTextModel(SubclassedBaseModel):
    """Gemma 3n text decoder backbone (no LM head).

    Google's on-device decoder: scaled embeddings feed a 4-stream **AltUp** state
    (magnitude-matched projections), and every block runs AltUp predict -> (input
    norm, **LAuReL** low-rank residual, GQA attention with per-head q/k/v norms) ->
    GeGLU MLP (Gaussian top-k **activation sparsity** on the early layers, and a
    per-layer **MatFormer** width) -> AltUp correct, then folds a **Per-Layer
    Embedding** through a gate/projection into the non-active streams. A 5:1
    sliding/global schedule (``layer_types``) with dual rotary bases, and a tail of
    ``num_kv_shared_layers`` that reuse an earlier layer's K/V per attention type.
    The streams are magnitude-matched back to one and normed at the end. Returns raw
    features; use :class:`Gemma3nGenerate`.

    Args:
        vocab_size / embed_dim / mlp_dim / num_layers / num_heads / num_kv_heads /
        head_dim: Core decoder geometry (``mlp_dim`` may be a per-layer list).
        sliding_window / sliding_window_pattern / layer_types: Attention schedule.
        final_logit_softcapping / norm_eps / rope_theta / rope_local_theta: Misc.
        tie_embeddings: Whether :class:`Gemma3nGenerate` ties the LM head.
        vocab_size_per_layer_input / hidden_size_per_layer_input: Per-Layer Embeddings.
        altup_num_inputs / altup_active_idx / altup_correct_scale: AltUp settings.
        num_kv_shared_layers / laurel_rank / activation_sparsity_pattern: Extras.
    """

    HF_MODEL_TYPE = ("gemma3n", "gemma3n_text")
    config_class = Gemma3nTextConfig
    default_load_dtype = "bfloat16"  # Google ships gemma-3n in bf16

    def __init__(
        self,
        vocab_size=262400,
        embed_dim=2048,
        mlp_dim=16384,
        num_layers=35,
        num_heads=8,
        num_kv_heads=2,
        head_dim=256,
        sliding_window=512,
        sliding_window_pattern=5,
        layer_types=None,
        final_logit_softcapping=30.0,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_local_theta=10000.0,
        hidden_activation="gelu_pytorch_tanh",
        tie_embeddings=True,
        vocab_size_per_layer_input=262144,
        hidden_size_per_layer_input=256,
        altup_num_inputs=4,
        altup_active_idx=0,
        altup_coef_clip=120.0,
        altup_correct_scale=True,
        num_kv_shared_layers=15,
        laurel_rank=64,
        activation_sparsity_pattern=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.final_logit_softcapping = final_logit_softcapping
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.rope_local_theta = rope_local_theta
        self.tie_embeddings = tie_embeddings
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.altup_num_inputs = altup_num_inputs
        self.altup_active_idx = altup_active_idx
        self.altup_coef_clip = altup_coef_clip
        self.altup_correct_scale = altup_correct_scale
        self.num_kv_shared_layers = num_kv_shared_layers
        self.laurel_rank = laurel_rank
        self.first_kv_shared = num_layers - num_kv_shared_layers

        self.mlp_dims = (
            list(mlp_dim)
            if isinstance(mlp_dim, (list, tuple))
            else [mlp_dim] * num_layers
        )
        self.layer_types = (
            list(layer_types)
            if layer_types
            else [
                "full_attention"
                if (i + 1) % sliding_window_pattern == 0
                else "sliding_attention"
                for i in range(num_layers)
            ]
        )
        if activation_sparsity_pattern is not None:
            self.activation_sparsity_pattern = list(activation_sparsity_pattern)
        else:
            n_sparse = 10 if num_layers > 10 else 0
            self.activation_sparsity_pattern = [0.95] * n_sparse + [0.0] * (
                num_layers - n_sparse
            )

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.embed_tokens_per_layer = layers.Embedding(
            vocab_size_per_layer_input,
            num_layers * hidden_size_per_layer_input,
            name="embed_tokens_per_layer",
        )
        self.per_layer_model_projection = layers.Dense(
            num_layers * hidden_size_per_layer_input,
            use_bias=False,
            name="per_layer_model_projection",
        )
        self.per_layer_projection_norm = Gemma3nRMSNorm(
            eps=norm_eps, name="per_layer_projection_norm"
        )
        self.altup_projections = [
            layers.Dense(embed_dim, use_bias=False, name=f"altup_projection_{i}")
            for i in range(altup_num_inputs - 1)
        ]
        self.altup_unembed_projections = [
            layers.Dense(
                embed_dim, use_bias=False, name=f"altup_unembed_projection_{i}"
            )
            for i in range(altup_num_inputs - 1)
        ]

        self.decoder_layers = []
        for i in range(num_layers):
            is_shared = (
                num_kv_shared_layers > 0
                and self.first_kv_shared > 0
                and i >= self.first_kv_shared
            )
            self.decoder_layers.append(
                Gemma3nDecoderLayer(
                    embed_dim,
                    self.mlp_dims[i],
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    hidden_size_per_layer_input,
                    laurel_rank,
                    altup_num_inputs,
                    altup_active_idx,
                    altup_correct_scale,
                    activation_sparsity=self.activation_sparsity_pattern[i],
                    is_kv_shared=is_shared,
                    norm_eps=norm_eps,
                    name=f"decoder_layer_{i}",
                )
            )
        self.final_norm = Gemma3nRMSNorm(eps=norm_eps, name="final_norm")

    def is_sliding(self, layer_idx):
        return self.layer_types[layer_idx] == "sliding_attention"

    def embed_scaled(self, input_ids):
        return self.token_embedding(input_ids) * ops.cast(
            self.embed_dim**0.5, self.compute_dtype
        )

    def get_per_layer_inputs(self, tokens):
        # Per-Layer Embedding lookup (the hidden**0.5 scale is applied here since the
        # Keras Embedding is unscaled). ``tokens`` must be in [0, vocab_size_per_layer).
        b = ops.shape(tokens)[0]
        s = ops.shape(tokens)[1]
        ple = self.embed_tokens_per_layer(tokens) * ops.cast(
            self.hidden_size_per_layer_input**0.5, self.compute_dtype
        )
        return ops.reshape(
            ple, (b, s, self.num_layers, self.hidden_size_per_layer_input)
        )

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs):
        b = ops.shape(inputs_embeds)[0]
        s = ops.shape(inputs_embeds)[1]
        proj = self.per_layer_model_projection(inputs_embeds) * ops.cast(
            self.embed_dim**-0.5, self.compute_dtype
        )
        proj = ops.reshape(
            proj, (b, s, self.num_layers, self.hidden_size_per_layer_input)
        )
        proj = self.per_layer_projection_norm(proj)
        return (proj + per_layer_inputs) * ops.cast(2.0**-0.5, self.compute_dtype)

    def mask_per_layer_tokens(self, input_ids):
        valid = ops.logical_and(
            input_ids >= 0, input_ids < self.vocab_size_per_layer_input
        )
        return ops.where(valid, input_ids, ops.zeros_like(input_ids))

    def compute_per_layer_inputs(self, input_ids, inputs_embeds):
        ple = self.get_per_layer_inputs(self.mask_per_layer_tokens(input_ids))
        return self.project_per_layer_inputs(inputs_embeds, ple)

    def altup_expand(self, hidden_0):
        target = ops.sqrt(ops.mean(ops.square(hidden_0), axis=-1, keepdims=True))
        streams = [hidden_0]
        for proj in self.altup_projections:
            cur = proj(hidden_0)
            mag = ops.sqrt(
                ops.maximum(ops.mean(ops.square(cur), axis=-1, keepdims=True), 1e-5)
            )
            streams.append(cur * target / mag)
        return ops.stack(streams, axis=0)  # (P, b, s, h)

    def altup_unembed(self, hidden):
        target = ops.sqrt(ops.mean(ops.square(hidden[0]), axis=-1, keepdims=True))
        streams = [hidden[0]]
        for i, proj in enumerate(self.altup_unembed_projections):
            cur = proj(hidden[i + 1])
            mag = ops.sqrt(
                ops.maximum(ops.mean(ops.square(cur), axis=-1, keepdims=True), 1e-5)
            )
            streams.append(cur * target / mag)
        return self.final_norm(ops.mean(ops.stack(streams, axis=0), axis=0))

    def compute_position_ids(self, attention_mask, batch, seq):
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            return ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        return ops.broadcast_to(ops.arange(seq), (batch, seq))

    def rope_tables(self, position_ids, theta):
        inv_freq = 1.0 / ops.power(
            theta, ops.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim
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
        within = ki > qi - self.sliding_window
        full = ops.cast(ops.where(causal, 0.0, MASK_NEG), "float32")[None, None]
        sliding = ops.cast(
            ops.where(ops.logical_and(causal, within), 0.0, MASK_NEG), "float32"
        )[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            pad = (1.0 - am)[:, None, None, :] * MASK_NEG
            full = full + pad
            sliding = sliding + pad
        return {"full_attention": full, "sliding_attention": sliding}

    def rope_for(self, position_ids):
        return {
            "full_attention": self.rope_tables(position_ids, self.rope_theta),
            "sliding_attention": self.rope_tables(position_ids, self.rope_local_theta),
        }

    def run_layers(self, hidden, rope, masks, per_layer_inputs):
        shared = {}
        for i, layer in enumerate(self.decoder_layers):
            lt = self.layer_types[i]
            cos, sin = rope[lt]
            pli = per_layer_inputs[:, :, i, :]
            is_shared = (
                self.num_kv_shared_layers > 0
                and self.first_kv_shared > 0
                and i >= self.first_kv_shared
            )
            hidden, kv = layer(
                hidden,
                cos,
                sin,
                pli,
                attention_mask=masks[lt],
                shared_kv=shared.get(lt) if is_shared else None,
            )
            if self.num_kv_shared_layers > 0 and not is_shared:
                shared[lt] = kv
        return hidden

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        attention_mask = inputs.get("attention_mask")
        inputs_embeds = inputs.get("inputs_embeds")
        per_layer_inputs = inputs.get("per_layer_inputs")
        if inputs_embeds is None:
            input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
            inputs_embeds = self.embed_scaled(input_ids)
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(
                    self.mask_per_layer_tokens(input_ids)
                )
        batch = int(inputs_embeds.shape[0])
        seq = int(inputs_embeds.shape[1])
        per_layer_inputs = self.project_per_layer_inputs(
            inputs_embeds, per_layer_inputs
        )
        hidden = self.altup_expand(inputs_embeds)
        position_ids = self.compute_position_ids(attention_mask, batch, seq)
        hidden = self.run_layers(
            hidden,
            self.rope_for(position_ids),
            self.build_masks(seq, attention_mask),
            per_layer_inputs,
        )
        return {"last_hidden_state": self.altup_unembed(hidden)}

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config.get("text_config", hf_config)
        return {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text["num_key_value_heads"],
            "head_dim": text.get("head_dim", 256),
            "sliding_window": text.get("sliding_window", 512),
            "layer_types": text.get("layer_types"),
            "final_logit_softcapping": text.get("final_logit_softcapping", 30.0),
            "norm_eps": text.get("rms_norm_eps", 1e-6),
            "rope_theta": (
                text.get("rope_parameters", {}).get("full_attention", {}) or {}
            ).get("rope_theta", 1000000.0),
            "rope_local_theta": (
                text.get("rope_parameters", {}).get("sliding_attention", {}) or {}
            ).get("rope_theta", 10000.0),
            "tie_embeddings": text.get("tie_word_embeddings", True),
            "vocab_size_per_layer_input": text.get(
                "vocab_size_per_layer_input", 262144
            ),
            "hidden_size_per_layer_input": text.get("hidden_size_per_layer_input", 256),
            "altup_num_inputs": text.get("altup_num_inputs", 4),
            "altup_active_idx": text.get("altup_active_idx", 0),
            "altup_coef_clip": text.get("altup_coef_clip", 120.0),
            "altup_correct_scale": text.get("altup_correct_scale", True),
            "num_kv_shared_layers": text.get("num_kv_shared_layers", 15),
            "laurel_rank": text.get("laurel_rank", 64),
            "activation_sparsity_pattern": text.get("activation_sparsity_pattern"),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma3n_hf_to_keras import transfer_gemma3n_weights

        transfer_gemma3n_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dims,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "sliding_window": self.sliding_window,
                "sliding_window_pattern": self.sliding_window_pattern,
                "layer_types": self.layer_types,
                "final_logit_softcapping": self.final_logit_softcapping,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "rope_local_theta": self.rope_local_theta,
                "tie_embeddings": self.tie_embeddings,
                "vocab_size_per_layer_input": self.vocab_size_per_layer_input,
                "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
                "altup_num_inputs": self.altup_num_inputs,
                "altup_active_idx": self.altup_active_idx,
                "altup_coef_clip": self.altup_coef_clip,
                "altup_correct_scale": self.altup_correct_scale,
                "num_kv_shared_layers": self.num_kv_shared_layers,
                "laurel_rank": self.laurel_rank,
                "activation_sparsity_pattern": self.activation_sparsity_pattern,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma3nGenerate(Gemma3nTextModel, BaseGeneration):
    """Gemma 3n text backbone + a (tied) LM head with fast ``.generate()``.

    The prefill runs the AltUp/LAuReL/PLE stack over the prompt and stores a
    fixed-size per-layer sliding/global K/V cache (the KV-shared tail reuses the last
    non-shared layer's cache per attention type); decoding steps one token through
    the same stack over that cache.
    """

    HF_MODEL_TYPE = ("gemma3n", "gemma3n_text")
    config_class = Gemma3nTextConfig
    default_load_dtype = "bfloat16"

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
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        token_ids = ops.cast(ops.convert_to_tensor(token_ids), "int32")
        inputs_embeds = self.embed_scaled(token_ids)
        per_layer_inputs = self.compute_per_layer_inputs(token_ids, inputs_embeds)
        hidden = self.altup_expand(inputs_embeds)
        position_ids = self.compute_position_ids(padding_mask, batch, prompt_len)
        rope = self.rope_for(position_ids)
        masks = self.build_masks(prompt_len, padding_mask)

        layer_caches = []
        shared = {}  # layer_type -> storing layer's prompt-length (k, v)
        shared_stacked = {}  # layer_type -> storing layer's padded [ck, cv]
        for i, layer in enumerate(self.decoder_layers):
            lt = self.layer_types[i]
            cos, sin = rope[lt]
            pli = per_layer_inputs[:, :, i, :]
            is_shared = (
                self.num_kv_shared_layers > 0
                and self.first_kv_shared > 0
                and i >= self.first_kv_shared
            )
            hidden, (k, v) = layer(
                hidden,
                cos,
                sin,
                pli,
                attention_mask=masks[lt],
                shared_kv=shared.get(lt) if is_shared else None,
            )
            if is_shared:
                layer_caches.append(shared_stacked[lt])
                continue
            nkv, hd = int(k.shape[1]), int(k.shape[3])
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            stacked = ops.stack([ck, cv], axis=1)
            layer_caches.append(stacked)
            if self.num_kv_shared_layers > 0:
                shared[lt] = (k, v)
                shared_stacked[lt] = stacked
        logits = self.project(self.altup_unembed(hidden)[:, -1, :])
        return tuple(layer_caches), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        batch = int(token_ids.shape[0])
        max_len = int(cache[0].shape[3])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        rope = self.rope_for(positions)
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
        km = {"full_attention": full_km, "sliding_attention": sliding_km}

        token_ids = ops.cast(token_ids, "int32")
        inputs_embeds = self.embed_scaled(token_ids)
        per_layer_inputs = self.compute_per_layer_inputs(token_ids, inputs_embeds)
        hidden = self.altup_expand(inputs_embeds)

        new_caches = []
        shared_stacked = {}
        for i, layer in enumerate(self.decoder_layers):
            lt = self.layer_types[i]
            cos, sin = rope[lt]
            pli = per_layer_inputs[:, :, i, :]
            is_shared = (
                self.num_kv_shared_layers > 0
                and self.first_kv_shared > 0
                and i >= self.first_kv_shared
            )
            if is_shared:
                stacked = shared_stacked[lt]
                hidden, _ = layer.decode_step(
                    hidden, cos, sin, pli, stacked[:, 0], stacked[:, 1], pos, km[lt]
                )
                new_caches.append(stacked)
                continue
            hidden, (ck, cv) = layer.decode_step(
                hidden, cos, sin, pli, cache[i][:, 0], cache[i][:, 1], pos, km[lt]
            )
            stacked = ops.stack([ck, cv], axis=1)
            new_caches.append(stacked)
            if self.num_kv_shared_layers > 0:
                shared_stacked[lt] = stacked
        logits = self.project(self.altup_unembed(hidden))[:, 0, :]
        return logits, tuple(new_caches)


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma3nAudioEncoder(layers.Layer):
    """USM audio encoder: SSCP sub-sampling then a conformer stack.

    Input mel features ``[B, T, F]`` (``F == input_feat_size``) plus an optional
    padding mask ``[B, T]`` (``True`` marks padded frames). Returns
    ``(audio_encodings [B, T', hidden], mask [B, T'])`` after temporal reduction."""

    def __init__(
        self,
        hidden_size=1536,
        input_feat_size=128,
        conf_num_hidden_layers=12,
        conf_num_attention_heads=8,
        conf_attention_chunk_size=12,
        conf_attention_context_left=13,
        conf_attention_context_right=0,
        conf_attention_logit_cap=50.0,
        conf_conv_kernel_size=5,
        conf_reduction_factor=4,
        conf_residual_weight=0.5,
        sscp_conv_channel_size=(128, 32),
        sscp_conv_kernel_size=((3, 3), (3, 3)),
        sscp_conv_stride_size=((2, 2), (2, 2)),
        sscp_conv_group_norm_eps=1e-3,
        rms_norm_eps=1e-6,
        gradient_clipping=1e10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.conf_reduction_factor = conf_reduction_factor
        self.sscp_conv_stride_size = [list(s) for s in sscp_conv_stride_size]
        self.subsample_conv_projection = Gemma3nAudioSubSampleConvProjection(
            input_feat_size,
            sscp_conv_channel_size,
            sscp_conv_kernel_size,
            sscp_conv_stride_size,
            hidden_size,
            norm_eps=sscp_conv_group_norm_eps,
            name="subsample_conv_projection",
        )
        self.conformer = [
            Gemma3nAudioConformerBlock(
                hidden_size,
                conf_num_attention_heads,
                conf_attention_chunk_size,
                conf_attention_context_left,
                conf_attention_context_right,
                conf_conv_kernel_size,
                conf_attention_logit_cap,
                rms_norm_eps,
                conf_residual_weight,
                gradient_clipping,
                name=f"conformer_{i}",
            )
            for i in range(conf_num_hidden_layers)
        ]

    def call(self, audio_mel, audio_mel_mask=None):
        x = self.subsample_conv_projection(audio_mel)  # [B, T_sub, D]
        t_sub = int(x.shape[1])
        time_stride = 1
        for s in self.sscp_conv_stride_size:
            time_stride *= s[0]

        if audio_mel_mask is None:
            audio_mel_mask = ops.zeros(
                (ops.shape(audio_mel)[0], ops.shape(audio_mel)[1]), dtype="bool"
            )
        max_idx = ops.shape(audio_mel_mask)[1] - 1
        indices = ops.minimum(ops.arange(t_sub) * time_stride, max_idx)
        indices = ops.broadcast_to(indices[None], (ops.shape(audio_mel_mask)[0], t_sub))
        mask = ops.take_along_axis(audio_mel_mask, indices, axis=1)  # [B, T_sub]

        for block in self.conformer:
            x = block(x, mask)

        if self.conf_reduction_factor > 1:
            x = x[:, :: self.conf_reduction_factor]
            mask = mask[:, :: self.conf_reduction_factor]
        x = ops.where(mask[..., None], ops.zeros_like(x), x)
        return x, mask


@keras.saving.register_keras_serializable(package="kerasformers")
class MobileNetV5Encoder(layers.Layer):
    """timm ``mobilenetv5_300m_enc`` in pure Keras (channels-last).

    Stem (Conv + RmsNormAct) then four stages of EdgeResidual / Universal Inverted
    Residual / MobileAttention blocks, whose last two stage outputs feed a
    Multi-Scale Fusion Adapter that emits a fixed ``output_resolution`` feature map.
    Input ``[B, H, W, 3]`` (pixels normalized to ``[-1, 1]``); output
    ``[B, 16, 16, hidden_size]`` (256 vision soft tokens after flatten)."""

    def __init__(self, hidden_size=2048, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.eps = eps
        arch = MNV5_ARCH
        stem = arch["stem"]
        self.conv_stem = ConvNormAct(
            stem["out"],
            stem["k"],
            stem["s"],
            False,
            True,
            bias=stem["b"],
            eps=eps,
            name="conv_stem",
        )
        self.stages = []
        for si, stage in enumerate(arch["stages"]):
            blocks = [
                build_block(spec, eps, f"blocks_{si}_{bi}")
                for bi, spec in enumerate(stage)
            ]
            self.stages.append(blocks)
        self.msfa_indices = arch["msfa"]["indices"]
        self.msfa = MobileNetV5MSFA(arch["msfa"], eps=eps, name="msfa")

    def call(self, pixel_values):
        x = self.conv_stem(pixel_values)
        # feature_info maps: index 1 = stage0 out, ..., index i+1 = stage i out.
        feats = [None]  # index 0 is the stem (unused by msfa)
        for blocks in self.stages:
            for block in blocks:
                x = block(x)
            feats.append(x)
        msfa_inputs = [feats[i] for i in self.msfa_indices]
        return self.msfa(msfa_inputs)  # [B, 16, 16, hidden_size]

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma3nModel(SubclassedBaseModel):
    """Gemma 3n vision + audio + text backbone (no LM head).

    Composes the MobileNet-V5 vision tower, the USM audio tower
    (:class:`Gemma3nAudioEncoder`), two soft-token projectors
    (:class:`Gemma3nMultimodalEmbedder`) and the text decoder
    (:class:`Gemma3nTextModel`). Hard multimodal token ids (the per-modality 128
    vocab, offset past the text vocab) are embedded via the projectors; pixel /
    audio inputs become soft tokens that are scattered onto the ``image_token_id``
    / ``audio_token_id`` slots. Returns raw text features; the LM head lives in
    :class:`Gemma3nForConditionalGeneration`.

    Args:
        text_config: Keyword arguments forwarded to :class:`Gemma3nTextModel`.
        vision_config: MobileNet-V5 tower config, or ``None`` for no vision tower.
        audio_config: USM tower config, or ``None`` for no audio tower.
        image_token_id / audio_token_id: Prompt slots that receive soft tokens.
        vision_soft_tokens_per_image / audio_soft_tokens_per_image: Soft-token counts.
    """

    HF_MODEL_TYPE = ("gemma3n",)
    config_class = None  # set below to Gemma3nConfig
    default_load_dtype = "bfloat16"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        image_token_id=262145,
        audio_token_id=262273,
        boi_token_id=255999,
        eoi_token_id=262144,
        boa_token_id=256000,
        eoa_token_id=262272,
        vision_soft_tokens_per_image=256,
        audio_soft_tokens_per_image=188,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_config = dict(text_config or {})
        self.vision_config = dict(vision_config) if vision_config else None
        self.audio_config = dict(audio_config) if audio_config else None
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_id = eoa_token_id
        self.vision_soft_tokens_per_image = vision_soft_tokens_per_image
        self.audio_soft_tokens_per_image = audio_soft_tokens_per_image
        self.tie_word_embeddings = tie_word_embeddings

        self.language_model = Gemma3nTextModel(
            **self.text_config, name="language_model"
        )
        text_hidden = self.language_model.embed_dim

        self.vision_tower = None
        self.embed_vision = None
        self.vision_vocab_offset = None
        self.vision_vocab_size = None
        if self.vision_config is not None:
            v = dict(self.vision_config)
            self.vision_vocab_size = v.pop("vocab_size", 128)
            self.vision_vocab_offset = v.pop("vocab_offset", 262144)
            v_hidden = v.get("hidden_size", 2048)
            v_eps = v.pop("rms_norm_eps", 1e-6)
            v.pop("architecture", None)
            v.pop("do_pooling", None)
            self.vision_tower = MobileNetV5Encoder(**v, name="vision_tower")
            self.embed_vision = Gemma3nMultimodalEmbedder(
                v_hidden,
                text_hidden,
                self.vision_vocab_size,
                self.vision_vocab_offset,
                eps=v_eps,
                name="embed_vision",
            )

        self.audio_tower = None
        self.embed_audio = None
        self.audio_vocab_offset = None
        self.audio_vocab_size = None
        if self.audio_config is not None:
            a = dict(self.audio_config)
            self.audio_vocab_size = a.pop("vocab_size", 128)
            self.audio_vocab_offset = a.pop("vocab_offset", 262272)
            a_hidden = a.get("hidden_size", 1536)
            a_eps = a.get("rms_norm_eps", 1e-6)
            self.audio_tower = Gemma3nAudioEncoder(**a, name="audio_tower")
            self.embed_audio = Gemma3nMultimodalEmbedder(
                a_hidden,
                text_hidden,
                self.audio_vocab_size,
                self.audio_vocab_offset,
                eps=a_eps,
                name="embed_audio",
            )

    def scatter_soft_tokens(self, text_embeds, slot_mask, features):
        # Fill every True position of slot_mask (row-major over batch then sequence)
        # with successive rows of features, mirroring HF's masked_scatter.
        shape = ops.shape(text_embeds)
        flat_mask = ops.reshape(slot_mask, (-1,))
        rank = ops.cumsum(ops.cast(flat_mask, "int32")) - 1
        rank = ops.clip(rank, 0, ops.shape(features)[0] - 1)
        gathered = ops.take(features, rank, axis=0)
        gathered = ops.reshape(gathered, shape)
        return ops.where(ops.expand_dims(slot_mask, -1), gathered, text_embeds)

    def get_image_features(self, pixel_values):
        feat = self.vision_tower(ops.convert_to_tensor(pixel_values))  # [B, H, W, C]
        b = ops.shape(feat)[0]
        v_hidden = self.embed_vision.multimodal_hidden_size
        feat = ops.reshape(feat, (b, self.vision_soft_tokens_per_image, v_hidden))
        feat = feat * ops.cast(v_hidden**0.5, feat.dtype)
        return self.embed_vision(inputs_embeds=feat)  # [B, soft, text_hidden]

    def get_audio_features(self, input_features, input_features_mask):
        mel_mask = (
            None
            if input_features_mask is None
            else ops.logical_not(ops.cast(input_features_mask, "bool"))
        )
        audio_out, audio_mask = self.audio_tower(
            ops.convert_to_tensor(input_features), mel_mask
        )
        a_feat = self.embed_audio(inputs_embeds=audio_out)  # [B, T', text_hidden]
        pad_id = self.audio_vocab_offset + self.audio_vocab_size - 1
        pad_emb = self.embed_audio(
            input_ids=ops.full((1, 1), pad_id, dtype="int32")
        )  # [1, 1, text_hidden]
        a_feat = ops.where(audio_mask[..., None], pad_emb, a_feat)
        b = ops.shape(a_feat)[0]
        t_prime = int(a_feat.shape[1])
        extra = self.audio_soft_tokens_per_image - t_prime
        if extra > 0:
            th = ops.shape(a_feat)[2]
            extra_feat = ops.broadcast_to(pad_emb, (b, extra, th))
            a_feat = ops.concatenate([a_feat, extra_feat], axis=1)
        else:
            a_feat = a_feat[:, : self.audio_soft_tokens_per_image]
        return a_feat  # [B, audio_soft_tokens_per_image, text_hidden]

    def fuse_embeds(
        self,
        input_ids,
        pixel_values=None,
        input_features=None,
        input_features_mask=None,
    ):
        lm = self.language_model
        ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        inputs_embeds = lm.embed_scaled(ids)
        per_layer_inputs = lm.get_per_layer_inputs(lm.mask_per_layer_tokens(ids))

        # Hard multimodal token ids embedded through the projectors' own tables.
        if self.embed_vision is not None:
            vision_end = self.vision_vocab_offset + self.vision_vocab_size
            vis_mask = ops.logical_and(
                ids >= self.vision_vocab_offset, ids < vision_end
            )
            dummy = self.vision_vocab_offset + self.vision_vocab_size - 1
            v_ids = ops.where(vis_mask, ids, dummy)
            v_emb = ops.cast(self.embed_vision(input_ids=v_ids), inputs_embeds.dtype)
            inputs_embeds = ops.where(vis_mask[..., None], v_emb, inputs_embeds)
        if self.embed_audio is not None:
            aud_mask = ids >= self.audio_vocab_offset
            dummy_a = self.audio_vocab_offset + self.audio_vocab_size - 1
            a_ids = ops.where(aud_mask, ids, dummy_a)
            a_emb = ops.cast(self.embed_audio(input_ids=a_ids), inputs_embeds.dtype)
            inputs_embeds = ops.where(aud_mask[..., None], a_emb, inputs_embeds)

        # Soft tokens scattered from actual pixel / audio inputs.
        if pixel_values is not None and self.vision_tower is not None:
            img_feat = self.get_image_features(pixel_values)
            th = ops.shape(img_feat)[-1]
            features = ops.cast(ops.reshape(img_feat, (-1, th)), inputs_embeds.dtype)
            inputs_embeds = self.scatter_soft_tokens(
                inputs_embeds, ids == self.image_token_id, features
            )
        if input_features is not None and self.audio_tower is not None:
            aud_feat = self.get_audio_features(input_features, input_features_mask)
            th = ops.shape(aud_feat)[-1]
            features = ops.cast(ops.reshape(aud_feat, (-1, th)), inputs_embeds.dtype)
            inputs_embeds = self.scatter_soft_tokens(
                inputs_embeds, ids == self.audio_token_id, features
            )
        return inputs_embeds, per_layer_inputs

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        inputs_embeds, per_layer_inputs = self.fuse_embeds(
            inputs["input_ids"],
            inputs.get("pixel_values"),
            inputs.get("input_features"),
            inputs.get("input_features_mask"),
        )
        return self.language_model(
            {
                "inputs_embeds": inputs_embeds,
                "per_layer_inputs": per_layer_inputs,
                "attention_mask": inputs.get("attention_mask"),
            }
        )

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config["text_config"]
        vision = hf_config.get("vision_config")
        audio = hf_config.get("audio_config")
        vision_ok = bool(vision) and vision.get("model_type") == "gemma3n_vision"
        audio_ok = bool(audio) and audio.get("model_type") == "gemma3n_audio"
        return {
            "text_config": Gemma3nTextModel.config_from_hf(hf_config),
            "vision_config": cls.vision_config_from_hf(vision) if vision_ok else None,
            "audio_config": cls.audio_config_from_hf(audio) if audio_ok else None,
            "image_token_id": hf_config.get("image_token_id", 262145),
            "audio_token_id": hf_config.get("audio_token_id", 262273),
            "boi_token_id": hf_config.get("boi_token_id", 255999),
            "eoi_token_id": hf_config.get("eoi_token_id", 262144),
            "boa_token_id": hf_config.get("boa_token_id", 256000),
            "eoa_token_id": hf_config.get("eoa_token_id", 262272),
            "vision_soft_tokens_per_image": hf_config.get(
                "vision_soft_tokens_per_image", 256
            ),
            "audio_soft_tokens_per_image": hf_config.get(
                "audio_soft_tokens_per_image", 188
            ),
            "tie_word_embeddings": text.get("tie_word_embeddings", True),
        }

    @staticmethod
    def vision_config_from_hf(vision):
        return {
            "architecture": vision.get("architecture", "mobilenetv5_300m_enc"),
            "hidden_size": vision.get("hidden_size", 2048),
            "vocab_size": vision.get("vocab_size", 128),
            "vocab_offset": vision.get("vocab_offset", 262144),
            "rms_norm_eps": vision.get("rms_norm_eps", 1e-6),
        }

    @staticmethod
    def audio_config_from_hf(audio):
        keys = (
            "vocab_size",
            "vocab_offset",
            "input_feat_size",
            "hidden_size",
            "rms_norm_eps",
            "gradient_clipping",
            "conf_attention_chunk_size",
            "conf_attention_context_left",
            "conf_attention_context_right",
            "conf_attention_logit_cap",
            "conf_num_attention_heads",
            "conf_num_hidden_layers",
            "conf_conv_kernel_size",
            "conf_reduction_factor",
            "conf_residual_weight",
            "sscp_conv_channel_size",
            "sscp_conv_group_norm_eps",
            "sscp_conv_kernel_size",
            "sscp_conv_stride_size",
        )
        return {k: audio[k] for k in keys if k in audio}

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma3n_hf_to_keras import transfer_gemma3n_weights

        transfer_gemma3n_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "text_config": self.text_config,
                "vision_config": self.vision_config,
                "audio_config": self.audio_config,
                "image_token_id": self.image_token_id,
                "audio_token_id": self.audio_token_id,
                "boi_token_id": self.boi_token_id,
                "eoi_token_id": self.eoi_token_id,
                "boa_token_id": self.boa_token_id,
                "eoa_token_id": self.eoa_token_id,
                "vision_soft_tokens_per_image": self.vision_soft_tokens_per_image,
                "audio_soft_tokens_per_image": self.audio_soft_tokens_per_image,
                "tie_word_embeddings": self.tie_word_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma3nForConditionalGeneration(Gemma3nModel, BaseGeneration):
    """Gemma 3n multimodal backbone + a (tied) LM head with fast ``.generate()``.

    The single generation entry point, like transformers'
    ``Gemma3nForConditionalGeneration``: it drives text-only and vision / audio
    prompts through one API. The prefill fuses soft tokens; decoding is text-only
    and reuses the per-layer sliding / global K/V cache. Pass ``pixel_values`` /
    ``input_features`` / ``input_features_mask`` as keyword prefill inputs to
    ``generate`` when the checkpoint has the towers."""

    HF_MODEL_TYPE = ("gemma3n", "gemma3n_text")
    config_class = None  # set below to Gemma3nConfig
    default_load_dtype = "bfloat16"

    eos_token_id = (1, 106)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lm = self.language_model
        self.lm_head = (
            None
            if self.tie_word_embeddings
            else layers.Dense(lm.vocab_size, use_bias=False, name="lm_head")
        )

    def project(self, hidden):
        lm = self.language_model
        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = ops.matmul(hidden, ops.transpose(lm.token_embedding.embeddings))
        if lm.final_logit_softcapping is not None:
            cap = lm.final_logit_softcapping
            logits = ops.tanh(logits / cap) * cap
        return logits

    def call(self, inputs):
        hidden = super().call(inputs)["last_hidden_state"]
        return {"logits": self.project(hidden), "last_hidden_state": hidden}

    def build_cache(
        self,
        token_ids,
        padding_mask,
        max_len,
        pixel_values=None,
        input_features=None,
        input_features_mask=None,
    ):
        lm = self.language_model
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        inputs_embeds, ple = self.fuse_embeds(
            token_ids, pixel_values, input_features, input_features_mask
        )
        per_layer_inputs = lm.project_per_layer_inputs(inputs_embeds, ple)
        hidden = lm.altup_expand(inputs_embeds)
        position_ids = lm.compute_position_ids(padding_mask, batch, prompt_len)
        rope = lm.rope_for(position_ids)
        masks = lm.build_masks(prompt_len, padding_mask)

        layer_caches = []
        shared = {}
        shared_stacked = {}
        for i, layer in enumerate(lm.decoder_layers):
            lt = lm.layer_types[i]
            cos, sin = rope[lt]
            pli = per_layer_inputs[:, :, i, :]
            is_shared = (
                lm.num_kv_shared_layers > 0
                and lm.first_kv_shared > 0
                and i >= lm.first_kv_shared
            )
            hidden, (k, v) = layer(
                hidden,
                cos,
                sin,
                pli,
                attention_mask=masks[lt],
                shared_kv=shared.get(lt) if is_shared else None,
            )
            if is_shared:
                layer_caches.append(shared_stacked[lt])
                continue
            nkv, hd = int(k.shape[1]), int(k.shape[3])
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            stacked = ops.stack([ck, cv], axis=1)
            layer_caches.append(stacked)
            if lm.num_kv_shared_layers > 0:
                shared[lt] = (k, v)
                shared_stacked[lt] = stacked
        logits = self.project(lm.altup_unembed(hidden)[:, -1, :])
        return tuple(layer_caches), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        # Decoding is text-only; delegate to the text stack's cached step.
        lm = self.language_model
        max_len = int(cache[0].shape[3])
        pos = cache_update_index
        batch = int(token_ids.shape[0])
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        rope = lm.rope_for(positions)
        ar = ops.arange(max_len)
        full_km = ops.cast(ops.where(ar <= pos, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        sliding_km = ops.cast(
            ops.where(
                ops.logical_and(ar <= pos, ar > pos - lm.sliding_window),
                0.0,
                MASK_NEG,
            ),
            "float32",
        )[None, None, None, :]
        km = {"full_attention": full_km, "sliding_attention": sliding_km}

        ids = ops.cast(token_ids, "int32")
        inputs_embeds = lm.embed_scaled(ids)
        per_layer_inputs = lm.compute_per_layer_inputs(ids, inputs_embeds)
        hidden = lm.altup_expand(inputs_embeds)

        new_caches = []
        shared_stacked = {}
        for i, layer in enumerate(lm.decoder_layers):
            lt = lm.layer_types[i]
            cos, sin = rope[lt]
            pli = per_layer_inputs[:, :, i, :]
            is_shared = (
                lm.num_kv_shared_layers > 0
                and lm.first_kv_shared > 0
                and i >= lm.first_kv_shared
            )
            if is_shared:
                stacked = shared_stacked[lt]
                hidden, _ = layer.decode_step(
                    hidden, cos, sin, pli, stacked[:, 0], stacked[:, 1], pos, km[lt]
                )
                new_caches.append(stacked)
                continue
            hidden, (ck, cv) = layer.decode_step(
                hidden, cos, sin, pli, cache[i][:, 0], cache[i][:, 1], pos, km[lt]
            )
            stacked = ops.stack([ck, cv], axis=1)
            new_caches.append(stacked)
            if lm.num_kv_shared_layers > 0:
                shared_stacked[lt] = stacked
        logits = self.project(lm.altup_unembed(hidden))[:, 0, :]
        return logits, tuple(new_caches)


Gemma3nModel.config_class = Gemma3nConfig
Gemma3nForConditionalGeneration.config_class = Gemma3nConfig
