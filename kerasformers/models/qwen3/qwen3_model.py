import keras
from keras import layers, ops

from kerasformers.base import SubclassedBaseModel

from .config import QWEN3_CONFIG, QWEN3_WEIGHTS
from .qwen3_layers import Qwen3DecoderLayer, Qwen3RMSNorm

_MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3Model(SubclassedBaseModel):
    """Qwen3 dense decoder-only transformer backbone (no LM head).

    ``token_embedding -> num_layers x Qwen3DecoderLayer -> final RMSNorm``, with
    grouped-query attention, per-head QK-norm (the reshaped query/key are RMSNorm'd
    before rotary), bias-free qkv projections, and 1D rotary positions. This is a
    subclassed (imperative) :class:`BaseModel`: the sequence length and decode-step
    count are data dependent, so the forward pass runs eagerly with ``keras.ops``
    rather than as a static graph. Returns raw features; use :class:`Qwen3Generate`
    for logits / text.

        model = Qwen3Model.from_weights("qwen3-0.6b")
        out = model({"input_ids": ids})["last_hidden_state"]  # (B, L, embed_dim)

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: SwiGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.
        norm_eps: RMSNorm epsilon (shared by the per-head QK-norms too).
        rope_theta: Rotary base frequency.
        tie_embeddings: Whether :class:`Qwen3Generate` ties the LM head to the
            token embedding instead of a separate projection.
    """

    HF_MODEL_TYPE = "qwen3"
    BASE_MODEL_CONFIG = QWEN3_CONFIG
    BASE_WEIGHT_CONFIG = QWEN3_WEIGHTS

    def __init__(
        self,
        vocab_size=151936,
        embed_dim=1024,
        mlp_dim=3072,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
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
            Qwen3DecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                self.head_dim,
                norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Qwen3RMSNorm(eps=norm_eps, name="final_norm")

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
        if attention_mask is not None:
            attn_mask = (
                attn_mask
                + (1.0 - ops.cast(am, "float32"))[:, None, None, :] * _MASK_NEG
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
            "num_kv_heads": hf_config["num_key_value_heads"],
            "head_dim": hf_config.get("head_dim"),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            "tie_embeddings": hf_config.get("tie_word_embeddings", True),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_qwen3_hf_to_keras import transfer_qwen3_weights

        transfer_qwen3_weights(keras_model, hf_state_dict)

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
class Qwen3Generate(Qwen3Model):
    """Qwen3 backbone + a language-model head and greedy ``.generate()``.

    Adds a vocabulary projection on top of :class:`Qwen3Model`: a separate
    bias-free ``lm_head`` when ``tie_embeddings`` is ``False``, otherwise the
    (transposed) token embedding (weight tying). ``call`` returns both ``logits``
    ``(batch, seq, vocab_size)`` and ``last_hidden_state``; :meth:`generate` does
    greedy decoding with a KV cache. Constructor ``Args`` are inherited from
    :class:`Qwen3Model`.

        gen = Qwen3Generate.from_weights("qwen3-0.6b")
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

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def generate(
        self, input_ids, attention_mask=None, max_new_tokens=128, eos_token_id=(151645,)
    ):
        # A parallel prefill populates a pre-allocated fixed-size KV cache, then the
        # decode loop runs as a compiled ``keras.ops.while_loop`` over single-token
        # steps. Because the cache shape is constant, the whole per-token step (all
        # layers + head) is traced/fused once and reused for every token -- no
        # per-token Python dispatch, no growing-cache reallocation, no per-step host
        # sync (the eos early-stop is evaluated on-device in the loop condition).
        # Greedy: the generated tokens are identical to the plain eager loop (verified
        # token-for-token). Output is a fixed ``(batch, max_new_tokens)`` padded with
        # the eos id after a sequence finishes.
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch = int(input_ids.shape[0])
        prompt_len = int(input_ids.shape[1])
        max_len = prompt_len + max_new_tokens
        hd, nkv = self.head_dim, self.num_kv_heads

        inv_freq = 1.0 / ops.power(
            self.rope_theta, ops.arange(0, hd, 2, dtype="float32") / hd
        )
        angles = ops.arange(max_len, dtype="float32")[:, None] * inv_freq
        emb = ops.concatenate([angles, angles], axis=-1)
        # Cast to the compute dtype so the directly-called decode_step matches the
        # autocast that Layer.__call__ applies during prefill -> consistent cache
        # dtype (a no-op under the default float32 policy).
        cos_table = ops.cast(ops.cos(emb), self.compute_dtype)
        sin_table = ops.cast(ops.sin(emb), self.compute_dtype)

        # ---- prefill (parallel over the prompt) ----
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
            cos_p = ops.take(cos_table, position_ids, axis=0)
            sin_p = ops.take(sin_table, position_ids, axis=0)
        else:
            cos_p = ops.broadcast_to(
                cos_table[None, :prompt_len], (batch, prompt_len, hd)
            )
            sin_p = ops.broadcast_to(
                sin_table[None, :prompt_len], (batch, prompt_len, hd)
            )
        qi = ops.arange(prompt_len)[:, None]
        ki = ops.arange(prompt_len)[None, :]
        causal = ops.cast(ops.where(ki <= qi, 0.0, _MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            causal = (
                causal + (1.0 - ops.cast(am, "float32"))[:, None, None, :] * _MASK_NEG
            )

        hidden = self.token_embedding(input_ids)
        cache_k, cache_v = [], []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden, cos_p, sin_p, attention_mask=causal, use_cache=True
            )
            cache_k.append(
                ops.slice_update(
                    ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
                )
            )
            cache_v.append(
                ops.slice_update(
                    ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
                )
            )
        last = self.final_norm(hidden)[:, -1:, :]
        first_tok = ops.cast(ops.argmax(self.project(last), axis=-1), "int32")

        eos = [
            int(e)
            for e in (
                eos_token_id
                if isinstance(eos_token_id, (list, tuple))
                else [eos_token_id]
            )
        ]
        first_eos = eos[0] if eos else 0
        if max_new_tokens <= 1:
            return ops.convert_to_numpy(first_tok)

        finished = ops.zeros((batch,), dtype="bool")
        for e in eos:
            finished = ops.logical_or(finished, first_tok[:, 0] == e)

        cache_k = ops.stack(cache_k, axis=0)  # (num_layers, B, nkv, max_len, hd)
        cache_v = ops.stack(cache_v, axis=0)
        key_positions = ops.arange(max_len)
        steps = max_new_tokens - 1
        # Tokens 1..steps; pre-filled with the eos id so an early stop leaves eos
        # padding and the output is a fixed (B, max_new_tokens).
        token_buffer = ops.full((steps, batch, 1), first_eos, dtype="int32")

        def cond(i, tok, ck, cv, pos, done, buf):
            return ops.logical_and(i < steps, ops.logical_not(ops.all(done)))

        def body(i, tok, ck, cv, pos, done, buf):
            cos_t = ops.broadcast_to(
                ops.take(cos_table, pos, axis=0)[None, None, :], (batch, 1, hd)
            )
            sin_t = ops.broadcast_to(
                ops.take(sin_table, pos, axis=0)[None, None, :], (batch, 1, hd)
            )
            key_mask = ops.cast(
                ops.where(key_positions <= pos, 0.0, _MASK_NEG), "float32"
            )[None, None, None, :]
            h = self.token_embedding(tok)
            new_k, new_v = [], []
            for j, layer in enumerate(self.decoder_layers):
                h, ck_j, cv_j = layer.decode_step(
                    h, cos_t, sin_t, ck[j], cv[j], pos, key_mask
                )
                new_k.append(ck_j)
                new_v.append(cv_j)
            h = self.final_norm(h)
            nxt = ops.cast(ops.argmax(self.project(h), axis=-1), "int32")
            nxt = ops.cast(ops.where(done[:, None], first_eos, nxt), "int32")
            for e in eos:
                done = ops.logical_or(done, nxt[:, 0] == e)
            buf = ops.slice_update(buf, (i, 0, 0), nxt[None])
            return (
                i + 1,
                nxt,
                ops.stack(new_k, 0),
                ops.stack(new_v, 0),
                pos + 1,
                done,
                buf,
            )

        init = (
            ops.convert_to_tensor(0, dtype="int32"),
            first_tok,
            cache_k,
            cache_v,
            ops.convert_to_tensor(prompt_len, dtype="int32"),
            finished,
            token_buffer,
        )
        token_buffer = ops.while_loop(cond, body, init, maximum_iterations=steps)[-1]
        tail = ops.transpose(token_buffer[:, :, 0], (1, 0))  # (B, steps)
        return ops.convert_to_numpy(ops.concatenate([first_tok, tail], axis=1))
