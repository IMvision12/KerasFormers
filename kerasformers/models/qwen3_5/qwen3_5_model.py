import keras
from keras import layers, ops

from kerasformers.base import SubclassedBaseModel
from kerasformers.base.constants import MASK_NEG

from .config import QWEN3_5_CONFIG, QWEN3_5_WEIGHTS
from .qwen3_5_layers import Qwen3_5DecoderLayer, Qwen3_5RMSNorm


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3_5Model(SubclassedBaseModel):
    """Qwen3.5 (Qwen3-Next) hybrid decoder-only backbone (no LM head).

    ``token_embedding -> num_layers x Qwen3_5DecoderLayer -> final RMSNorm``. The
    decoder layers alternate two token mixers: most are Gated-DeltaNet *linear
    attention* (conv1d + delta-rule recurrence), and every ``full_attention_interval``
    -th layer is *gated full attention* (GQA, QK-norm, partial rotary, sigmoid
    output gate). RMSNorm is zero-centered. This is a subclassed (imperative)
    :class:`BaseModel`: the forward pass runs eagerly with ``keras.ops``. Returns
    raw features; use :class:`Qwen3_5Generate` for logits / text.

        model = Qwen3_5Model.from_weights("hf:Qwen/Qwen3.5-...")
        out = model({"input_ids": ids})["last_hidden_state"]  # (B, L, embed_dim)

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: SwiGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads in the full-attention layers.
        num_kv_heads: Key/value heads in the full-attention layers (GQA).
        head_dim: Per-head dim of the full-attention layers.
        norm_eps: RMSNorm epsilon (shared everywhere, incl. QK-norm).
        rope_theta: Rotary base frequency.
        partial_rotary_factor: Fraction of ``head_dim`` that gets rotary
            (``rotary_dim = int(head_dim * partial_rotary_factor)``).
        tie_embeddings: Whether :class:`Qwen3_5Generate` ties the LM head to the
            token embedding instead of a separate projection.
        full_attention_interval: Place a full-attention layer every Nth block;
            all others are Gated-DeltaNet linear-attention layers.
        linear_conv_kernel_dim: Causal conv1d kernel width in the linear layers.
        linear_key_head_dim, linear_value_head_dim: Per-head dims of the linear
            attention key/value.
        linear_num_key_heads, linear_num_value_heads: Head counts of the linear
            attention key/value.
    """

    HF_MODEL_TYPE = ("qwen3_5", "qwen3_5_text")
    BASE_MODEL_CONFIG = QWEN3_5_CONFIG
    BASE_WEIGHT_CONFIG = QWEN3_5_WEIGHTS

    def __init__(
        self,
        vocab_size=248320,
        embed_dim=1024,
        mlp_dim=3584,
        num_layers=24,
        num_heads=8,
        num_kv_heads=2,
        head_dim=256,
        norm_eps=1e-6,
        rope_theta=10000000.0,
        partial_rotary_factor=0.25,
        tie_embeddings=True,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
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
        self.partial_rotary_factor = partial_rotary_factor
        self.tie_embeddings = tie_embeddings
        self.full_attention_interval = full_attention_interval
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.rotary_dim = int(head_dim * partial_rotary_factor)

        layer_cfg = {
            "embed_dim": embed_dim,
            "mlp_dim": mlp_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "rotary_dim": self.rotary_dim,
            "norm_eps": norm_eps,
            "linear_conv_kernel_dim": linear_conv_kernel_dim,
            "linear_key_head_dim": linear_key_head_dim,
            "linear_value_head_dim": linear_value_head_dim,
            "linear_num_key_heads": linear_num_key_heads,
            "linear_num_value_heads": linear_num_value_heads,
        }
        self.layer_types = [
            "full_attention"
            if (i + 1) % full_attention_interval == 0
            else "linear_attention"
            for i in range(num_layers)
        ]
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Qwen3_5DecoderLayer(
                layer_cfg, self.layer_types[i], name=f"decoder_layer_{i}"
            )
            for i in range(num_layers)
        ]
        self.final_norm = Qwen3_5RMSNorm(eps=norm_eps, name="final_norm")

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
            ops.arange(0, self.rotary_dim, 2, dtype="float32") / self.rotary_dim,
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        cos, sin = ops.cos(emb), ops.sin(emb)
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        attn_mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        pad_mask = None
        if attention_mask is not None:
            am_f = ops.cast(am, "float32")
            attn_mask = attn_mask + (1.0 - am_f)[:, None, None, :] * MASK_NEG
            pad_mask = am_f[:, :, None]
        for layer in self.decoder_layers:
            hidden = layer(
                hidden, cos, sin, attention_mask=attn_mask, pad_mask=pad_mask
            )
        return {"last_hidden_state": self.final_norm(hidden)}

    @classmethod
    def config_from_hf(cls, hf_config):
        c = hf_config.get("text_config", hf_config)
        rope = c.get("rope_parameters", c)
        return {
            "vocab_size": c["vocab_size"],
            "embed_dim": c["hidden_size"],
            "mlp_dim": c["intermediate_size"],
            "num_layers": c["num_hidden_layers"],
            "num_heads": c["num_attention_heads"],
            "num_kv_heads": c["num_key_value_heads"],
            "head_dim": c["head_dim"],
            "norm_eps": c.get("rms_norm_eps", 1e-6),
            "rope_theta": rope.get("rope_theta", c.get("rope_theta", 10000000.0)),
            "partial_rotary_factor": rope.get("partial_rotary_factor", 0.25),
            "tie_embeddings": c.get("tie_word_embeddings", True),
            "full_attention_interval": c["full_attention_interval"],
            "linear_conv_kernel_dim": c["linear_conv_kernel_dim"],
            "linear_key_head_dim": c["linear_key_head_dim"],
            "linear_value_head_dim": c["linear_value_head_dim"],
            "linear_num_key_heads": c["linear_num_key_heads"],
            "linear_num_value_heads": c["linear_num_value_heads"],
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_qwen3_5_hf_to_keras import transfer_qwen3_5_weights

        transfer_qwen3_5_weights(keras_model, hf_state_dict)

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
                "partial_rotary_factor": self.partial_rotary_factor,
                "tie_embeddings": self.tie_embeddings,
                "full_attention_interval": self.full_attention_interval,
                "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
                "linear_key_head_dim": self.linear_key_head_dim,
                "linear_value_head_dim": self.linear_value_head_dim,
                "linear_num_key_heads": self.linear_num_key_heads,
                "linear_num_value_heads": self.linear_num_value_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3_5Generate(Qwen3_5Model):
    """Qwen3.5 backbone + a language-model head and greedy ``.generate()``.

    Adds a vocabulary projection on top of :class:`Qwen3_5Model`: a separate
    bias-free ``lm_head`` when ``tie_embeddings`` is ``False``, otherwise the
    (transposed) token embedding (weight tying). ``call`` returns both ``logits``
    ``(batch, seq, vocab_size)`` and ``last_hidden_state``; :meth:`generate` does
    greedy decoding with a hybrid cache — a ``(key, value)`` tuple for the full-
    attention layers and a ``(conv_state, recurrent_state)`` tuple for the
    Gated-DeltaNet layers, carried per layer. Constructor ``Args`` are inherited
    from :class:`Qwen3_5Model`.

        gen = Qwen3_5Generate.from_weights("hf:Qwen/Qwen3.5-...")
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
        self, input_ids, attention_mask=None, max_new_tokens=128, eos_token_id=(248044,)
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
            ops.arange(0, self.rotary_dim, 2, dtype="float32") / self.rotary_dim,
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        cos, sin = ops.cos(emb), ops.sin(emb)
        qi = ops.arange(prompt_len)[:, None]
        ki = ops.arange(prompt_len)[None, :]
        causal = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        pad_mask = None
        if attention_mask is not None:
            am_f = ops.cast(am, "float32")
            causal = causal + (1.0 - am_f)[:, None, None, :] * MASK_NEG
            pad_mask = am_f[:, :, None]
        cache = []
        for layer in self.decoder_layers:
            hidden, state = layer(
                hidden,
                cos,
                sin,
                attention_mask=causal,
                use_cache=True,
                pad_mask=pad_mask,
            )
            cache.append(state)
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
                ops.arange(0, self.rotary_dim, 2, dtype="float32") / self.rotary_dim,
            )
            freqs = ops.cast(pos, "float32")[..., None] * inv_freq
            emb = ops.concatenate([freqs, freqs], axis=-1)
            c, s = ops.cos(emb), ops.sin(emb)
            hidden = self.token_embedding(next_tok)
            new_cache = []
            for i, layer in enumerate(self.decoder_layers):
                hidden, state = layer(
                    hidden, c, s, past_key_value=cache[i], use_cache=True
                )
                new_cache.append(state)
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
