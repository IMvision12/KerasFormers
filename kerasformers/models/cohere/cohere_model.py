import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .cohere_config import COHERE_CONFIG, COHERE_WEIGHTS_URLS
from .cohere_layers import CohereDecoderLayer, CohereLayerNorm

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class CohereModel(SubclassedBaseModel):
    """Cohere Command-R decoder backbone (no LM head).

    Parallel attention + MLP blocks off a single Cohere LayerNorm (mean-
    centered), interleaved rotary embeddings, optional per-head QK-norm, and a
    final LayerNorm. Returns raw features; use :class:`CohereGenerate` for
    logits (which applies ``logit_scale``).

    Args:
        vocab_size / embed_dim / num_layers / num_heads / num_kv_heads /
        head_dim: Geometry.
        mlp_dim: SwiGLU hidden width.
        use_qk_norm: Per-head QK LayerNorm (Command-R 08-2024 uses it).
        norm_eps: LayerNorm epsilon.
        rope_theta: Rotary base frequency.
        attention_bias: Attention projection bias.
        logit_scale: Output-logit multiplier (applied in the head).
        tie_embeddings: Whether the head ties to the token embedding.
    """

    HF_MODEL_TYPE = "cohere"
    BASE_MODEL_CONFIG = COHERE_CONFIG
    BASE_WEIGHT_CONFIG = COHERE_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=256000,
        embed_dim=8192,
        num_layers=40,
        num_heads=64,
        num_kv_heads=64,
        head_dim=None,
        mlp_dim=22528,
        use_qk_norm=False,
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
        self.use_qk_norm = use_qk_norm
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.logit_scale = logit_scale
        self.tie_embeddings = tie_embeddings

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            CohereDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                self.head_dim,
                use_qk_norm=use_qk_norm,
                norm_eps=norm_eps,
                attention_bias=attention_bias,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = CohereLayerNorm(eps=norm_eps, name="final_norm")

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

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def forward_features(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        hidden = self.token_embedding(input_ids)
        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables(position_ids)
        attn_mask = self.causal_mask(seq, inputs.get("attention_mask"))
        for layer in self.decoder_layers:
            hidden = layer(hidden, cos, sin, attention_mask=attn_mask)
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
            "head_dim": hf_config.get("head_dim"),
            "mlp_dim": hf_config["intermediate_size"],
            "use_qk_norm": bool(hf_config.get("use_qk_norm") or False),
            "norm_eps": hf_config.get("layer_norm_eps", 1e-5),
            "rope_theta": rope.get("rope_theta", hf_config.get("rope_theta", 10000.0)),
            "attention_bias": bool(hf_config.get("attention_bias") or False),
            "logit_scale": hf_config.get("logit_scale", 0.0625),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings", True)),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_cohere_hf_to_keras import transfer_cohere_weights

        transfer_cohere_weights(keras_model, hf_state_dict)

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
                "use_qk_norm": self.use_qk_norm,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "attention_bias": self.attention_bias,
                "logit_scale": self.logit_scale,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class CohereGenerate(CohereModel, BaseGeneration):
    """Cohere Command-R with a language-model head + fast ``.generate()``.

    Adds a vocabulary projection on top of :class:`CohereModel`: a bias-free
    ``lm_head`` when ``tie_embeddings`` is ``False``, otherwise the tied token
    embedding. Either way the logits are scaled by ``logit_scale`` (Cohere's
    output-temperature factor). ``call`` returns both ``logits`` and the final
    ``last_hidden_state``.

    Fast generation comes from :class:`~kerasformers.base.BaseGeneration`'s
    fixed-cache compiled decode loop: :meth:`build_cache` runs the prompt prefill
    once into a stacked per-layer KV cache, then :meth:`call_with_cache` performs
    the incremental single-token decode steps with a windowed key mask.
    ``eos_token_id`` defaults to Cohere's ``<|END_OF_TURN_TOKEN|>`` (255001);
    pass an explicit ``eos_token_id`` to :meth:`generate` to override.

    Construction mirrors :class:`CohereModel`::

        gen = CohereGenerate.from_weights("hf:CohereLabs/aya-expanse-8b")
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
        causal = self.causal_mask(prompt_len, padding_mask)
        hidden = self.token_embedding(ops.cast(token_ids, "int32"))
        layer_caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden, cos_p, sin_p, attention_mask=causal, use_cache=True
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
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos_t, sin_t, cache[:, i, 0], cache[:, i, 1], pos, key_mask
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
