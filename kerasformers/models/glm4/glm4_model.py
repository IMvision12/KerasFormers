import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .config import GLM4_CONFIG, GLM4_WEIGHTS_URLS
from .glm4_layers import Glm4DecoderLayer, Glm4RMSNorm

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4Model(SubclassedBaseModel):
    """GLM-4-0414 decoder backbone (no LM head).

    Same as GLM-4 but each block adds sandwich norms — the attention and MLP
    outputs are RMSNorm'd before being added back to the residual. Grouped-query
    attention, partial *interleaved* rotary, biased q/k/v, fused-SwiGLU MLP.
    Returns raw features; use :class:`Glm4Generate` for logits / text.

    Args:
        vocab_size / embed_dim / num_layers / num_heads / num_kv_heads /
        head_dim: Geometry.
        mlp_dim: SwiGLU hidden width (``intermediate_size``).
        partial_rotary_factor: Fraction of each head that receives rotary.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        attention_bias: Whether q/k/v carry bias.
        tie_embeddings: Whether the head ties to the token embedding.
    """

    HF_MODEL_TYPE = "glm4"
    BASE_MODEL_CONFIG = GLM4_CONFIG
    BASE_WEIGHT_CONFIG = GLM4_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=151552,
        embed_dim=4096,
        num_layers=40,
        num_heads=32,
        num_kv_heads=2,
        head_dim=128,
        mlp_dim=13696,
        partial_rotary_factor=0.5,
        norm_eps=0.00000015625,
        rope_theta=10000.0,
        attention_bias=True,
        tie_embeddings=False,
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
        self.partial_rotary_factor = partial_rotary_factor
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.tie_embeddings = tie_embeddings
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Glm4DecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                self.head_dim,
                self.rotary_dim,
                norm_eps=norm_eps,
                attention_bias=attention_bias,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Glm4RMSNorm(eps=norm_eps, name="final_norm")

    def rope_tables(self, position_ids):
        rd = self.rotary_dim
        inv_freq = 1.0 / ops.power(
            self.rope_theta, ops.arange(0, rd, 2, dtype="float32") / rd
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
        prf = rope.get("partial_rotary_factor")
        if prf is None:
            prf = hf_config.get("partial_rotary_factor", 0.5)
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
            "partial_rotary_factor": prf,
            "norm_eps": hf_config.get("rms_norm_eps", 0.00000015625),
            "rope_theta": rope.get("rope_theta", hf_config.get("rope_theta", 10000.0)),
            "attention_bias": bool(hf_config.get("attention_bias", True)),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings") or False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_glm4_hf_to_keras import transfer_glm4_weights

        transfer_glm4_weights(keras_model, hf_state_dict)

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
                "partial_rotary_factor": self.partial_rotary_factor,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "attention_bias": self.attention_bias,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4Generate(Glm4Model, BaseGeneration):
    """GLM-4-0414 with an LM head + fast ``.generate()``."""

    eos_token_id = (151329, 151336, 151338)

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
