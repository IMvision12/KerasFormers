import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .gemma_config import GEMMA_CONFIG, GEMMA_WEIGHTS_URLS
from .gemma_layers import GemmaDecoderLayer, GemmaRMSNorm

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class GemmaModel(SubclassedBaseModel):
    """Gemma decoder-only transformer backbone (no LM head).

    ``token_embedding * sqrt(embed_dim) -> num_layers x GemmaDecoderLayer ->
    final RMSNorm`` with the Gemma signature pieces: ``(1 + w)`` RMSNorm,
    GeGLU (tanh-approximate gelu) MLPs, scaled token embeddings, ``head_dim``
    256 decoupled from ``embed_dim // num_heads`` (the 2B is MQA with one
    K/V head), and a tied LM head. Subclassed (imperative) model; returns
    raw features: use :class:`GemmaGenerate` for logits / text.

        model = GemmaModel.from_weights("gemma-2b")
        out = model({"input_ids": ids})["last_hidden_state"]  # (B, L, embed_dim)

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: GeGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer.
        head_dim: Per-head dim.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        tie_embeddings: Whether :class:`GemmaGenerate` ties the LM head to the
            token embedding (Gemma checkpoints do).
    """

    HF_MODEL_TYPE = "gemma"
    BASE_MODEL_CONFIG = GEMMA_CONFIG
    BASE_WEIGHT_CONFIG = GEMMA_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=256000,
        embed_dim=2048,
        mlp_dim=16384,
        num_layers=18,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
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
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            GemmaDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = GemmaRMSNorm(eps=norm_eps, name="final_norm")

    def embed_scaled(self, input_ids):
        # Gemma multiplies token embeddings by sqrt(embed_dim).
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
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        attn_mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            attn_mask = (
                attn_mask + (1.0 - ops.cast(am, "float32"))[:, None, None, :] * MASK_NEG
            )
        for layer in self.decoder_layers:
            hidden = layer(hidden, cos, sin, attention_mask=attn_mask)
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
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "tie_embeddings": hf_config.get("tie_word_embeddings", True),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma_hf_to_keras import transfer_gemma_weights

        transfer_gemma_weights(keras_model, hf_state_dict)

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
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GemmaGenerate(GemmaModel, BaseGeneration):
    """Gemma backbone + a (tied) language-model head and fast ``.generate()``.

    Adds the vocabulary projection on top of :class:`GemmaModel`, the
    transposed scaled token embedding when ``tie_embeddings`` (all Gemma
    checkpoints), else a separate bias-free ``lm_head``. ``call`` returns both
    ``logits`` ``(batch, seq, vocab_size)`` and ``last_hidden_state``. Fast
    generation comes from :class:`~kerasformers.base.BaseGeneration` via
    ``build_cache`` / ``call_with_cache``. Constructor ``Args`` are inherited
    from :class:`GemmaModel`.

        gen = GemmaGenerate.from_weights("gemma-2b-it")
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
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def call(self, inputs):
        hidden = super().call(inputs)["last_hidden_state"]
        return {"logits": self.project(hidden), "last_hidden_state": hidden}

    def build_cache(self, token_ids, padding_mask, max_len):
        # Parallel prefill into a fixed (B, num_layers, 2, num_kv_heads,
        # max_len, head_dim) cache. Returns (cache, last-token logits).
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        if padding_mask is not None:
            am = ops.cast(padding_mask, "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos, sin = self.rope_tables(position_ids)
        qi = ops.arange(prompt_len)[:, None]
        ki = ops.arange(prompt_len)[None, :]
        causal = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if padding_mask is not None:
            causal = (
                causal + (1.0 - ops.cast(am, "float32"))[:, None, None, :] * MASK_NEG
            )
        hidden = self.embed_scaled(token_ids)
        layer_caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden, cos, sin, attention_mask=causal, use_cache=True
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
        # One decode step at position ``cache_update_index``.
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.embed_scaled(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[:, i, 0], cache[:, i, 1], pos, key_mask
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
