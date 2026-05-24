import keras
from keras import layers, ops

from kerasformers.base import BaseModel

from .config import QWEN2_CONFIG, QWEN2_WEIGHTS
from .qwen2_layers import Qwen2DecoderLayer, Qwen2RMSNorm

_MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2Model(BaseModel):
    """Qwen2 dense decoder-only transformer backbone (no LM head).

    ``token_embedding -> num_layers x Qwen2DecoderLayer -> final RMSNorm``, with
    grouped-query attention and 1D rotary positions. This is a subclassed
    (imperative) :class:`BaseModel`: the sequence length and decode-step count are
    data dependent, so the forward pass runs eagerly with ``keras.ops`` rather
    than as a static graph. Returns raw features; use :class:`Qwen2Generate` for
    logits / text.

        model = Qwen2Model.from_weights("qwen2-0.5b-instruct")
        out = model({"input_ids": ids})["last_hidden_state"]  # (B, L, embed_dim)

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: SwiGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        tie_embeddings: Whether :class:`Qwen2Generate` ties the LM head to the
            token embedding instead of a separate projection.
    """

    HF_MODEL_TYPE = "qwen2"
    BASE_MODEL_CONFIG = QWEN2_CONFIG
    BASE_WEIGHT_CONFIG = QWEN2_WEIGHTS

    def __init__(
        self,
        vocab_size=151936,
        embed_dim=896,
        mlp_dim=4864,
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        head_dim=None,
        norm_eps=1e-6,
        rope_theta=1000000.0,
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
        self.head_dim = head_dim or embed_dim // num_heads
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Qwen2DecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim=self.head_dim,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Qwen2RMSNorm(eps=norm_eps, name="final_norm")

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        attention_mask = inputs.get("attention_mask")
        hidden = self.token_embedding(input_ids)
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        inv_freq = 1.0 / ops.power(
            self.rope_theta,
            ops.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim,
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        cos, sin = ops.cos(emb), ops.sin(emb)
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        attn_mask = ops.cast(ops.where(ki <= qi, 0.0, _MASK_NEG), "float32")[None, None]
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
            "num_kv_heads": hf_config["num_key_value_heads"],
            "head_dim": hf_config.get("head_dim"),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            "tie_embeddings": hf_config.get("tie_word_embeddings", True),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_qwen2_hf_to_keras import transfer_qwen2_weights

        transfer_qwen2_weights(keras_model, hf_state_dict)

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
class Qwen2Generate(Qwen2Model):
    """Qwen2 backbone + a language-model head and greedy ``.generate()``.

    Adds a vocabulary projection on top of :class:`Qwen2Model`: a separate
    bias-free ``lm_head`` when ``tie_embeddings`` is ``False``, otherwise the
    (transposed) token embedding (weight tying). ``call`` returns both ``logits``
    ``(batch, seq, vocab_size)`` and ``last_hidden_state``; :meth:`generate` does
    greedy decoding with a KV cache. Constructor ``Args`` are inherited from
    :class:`Qwen2Model`.

        gen = Qwen2Generate.from_weights("qwen2-0.5b-instruct")
        ids = gen.generate(tokenizer(messages)["input_ids"])
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = (
            None
            if self.tie_embeddings
            else layers.Dense(self.vocab_size, use_bias=False, name="lm_head")
        )

    def call(self, inputs):
        hidden = super().call(inputs)["last_hidden_state"]
        logits = (
            self.lm_head(hidden)
            if self.lm_head is not None
            else ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))
        )
        return {"logits": logits, "last_hidden_state": hidden}

    def generate(
        self, input_ids, attention_mask=None, max_new_tokens=128, eos_token_id=(151645,)
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, prompt_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        hidden = self.token_embedding(input_ids)
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        inv_freq = 1.0 / ops.power(
            self.rope_theta,
            ops.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim,
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        cos, sin = ops.cos(emb), ops.sin(emb)
        qi = ops.arange(prompt_len)[:, None]
        ki = ops.arange(prompt_len)[None, :]
        causal = ops.cast(ops.where(ki <= qi, 0.0, _MASK_NEG), "float32")[None, None]
        cache = []
        for layer in self.decoder_layers:
            hidden, kv = layer(hidden, cos, sin, attention_mask=causal, use_cache=True)
            cache.append(kv)
        hidden = self.final_norm(hidden)[:, -1:, :]
        logits = (
            self.lm_head(hidden)
            if self.lm_head is not None
            else ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))
        )
        next_tok = ops.cast(ops.argmax(logits, axis=-1), "int32")

        eos = [
            int(e)
            for e in (
                eos_token_id
                if isinstance(eos_token_id, (list, tuple))
                else [eos_token_id]
            )
        ]
        first_eos = eos[0] if eos else 0
        finished = ops.zeros((batch,), dtype="bool")
        for e in eos:
            finished = ops.logical_or(finished, next_tok[:, 0] == e)
        generated = [next_tok]
        cur_len = prompt_len
        for _ in range(max_new_tokens - 1):
            if bool(ops.all(finished)):
                break
            pos = ops.full((batch, 1), cur_len, dtype="int32")
            inv_freq = 1.0 / ops.power(
                self.rope_theta,
                ops.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim,
            )
            freqs = ops.cast(pos, "float32")[..., None] * inv_freq
            emb = ops.concatenate([freqs, freqs], axis=-1)
            c, s = ops.cos(emb), ops.sin(emb)
            hidden = self.token_embedding(next_tok)
            new_cache = []
            for i, layer in enumerate(self.decoder_layers):
                hidden, kv = layer(
                    hidden, c, s, past_key_value=cache[i], use_cache=True
                )
                new_cache.append(kv)
            hidden = self.final_norm(hidden)
            cache = new_cache
            logits = (
                self.lm_head(hidden)
                if self.lm_head is not None
                else ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))
            )
            next_tok = ops.cast(ops.argmax(logits, axis=-1), "int32")
            next_tok = ops.cast(
                ops.where(finished[:, None], first_eos, next_tok), "int32"
            )
            generated.append(next_tok)
            cur_len += 1
            for e in eos:
                finished = ops.logical_or(finished, next_tok[:, 0] == e)
        return ops.convert_to_numpy(ops.concatenate(generated, axis=1))
