"""Qwen3.5 hybrid LLM (text backbone) in pure Keras 3 (self-contained).

``Qwen3_5Model`` returns features (``last_hidden_state``); ``Qwen3_5Generate``
adds the LM head + greedy ``.generate()``. Weights load on the fly from any
Qwen3.5 checkpoint's text tower (``model.language_model.*``):

    gen = Qwen3_5Generate.from_weights("hf:Qwen/Qwen3.5-0.8B")

Most layers are Gated-DeltaNet linear attention; every ``full_attention_interval``
-th layer is gated full attention with partial rotary. For pure text the three
M-RoPE position axes coincide, so rotary reduces to standard 1D partial rope.
"""

import keras
import numpy as np
from keras import layers, ops

from kerasformers.base import BaseModel

from .config import QWEN3_5_CONFIG
from .qwen3_5_layers import Qwen3_5DecoderLayer, Qwen3_5RMSNorm, rope_cos_sin

_MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3_5Model(BaseModel):
    """Qwen3.5 decoder: embed -> hybrid linear/full layers -> RMSNorm."""

    HF_MODEL_TYPE = ("qwen3_5", "qwen3_5_text")
    BASE_MODEL_CONFIG = QWEN3_5_CONFIG
    BASE_WEIGHT_CONFIG = None

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=1024,
        intermediate_size=3584,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        rms_norm_eps=1e-6,
        rope_theta=10000000.0,
        partial_rotary_factor=0.25,
        tie_word_embeddings=True,
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
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.tie_word_embeddings = tie_word_embeddings
        self.full_attention_interval = full_attention_interval
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.rotary_dim = int(head_dim * partial_rotary_factor)

        layer_cfg = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "rotary_dim": self.rotary_dim,
            "rms_norm_eps": rms_norm_eps,
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
            for i in range(num_hidden_layers)
        ]
        self.embed_tokens = layers.Embedding(
            vocab_size, hidden_size, name="embed_tokens"
        )
        self.decoder_layers = [
            Qwen3_5DecoderLayer(layer_cfg, self.layer_types[i], name=f"layers_{i}")
            for i in range(num_hidden_layers)
        ]
        self.norm = Qwen3_5RMSNorm(eps=rms_norm_eps, name="norm")

    def _causal_mask(self, q_len, kv_len, offset):
        qi = np.arange(q_len)[:, None] + offset
        ki = np.arange(kv_len)[None, :]
        mask = np.where(ki <= qi, 0.0, _MASK_NEG).astype("float32")
        return ops.convert_to_tensor(mask[None, None])

    def _positions(self, attention_mask, batch, seq):
        if attention_mask is not None:
            am = np.asarray(ops.convert_to_numpy(attention_mask))
            pos = np.cumsum(am, axis=-1) - 1
            return np.where(am == 0, 1, pos).astype("int64")
        return np.broadcast_to(np.arange(seq), (batch, seq)).astype("int64")

    def _run_decoder(
        self,
        inputs_embeds,
        cos,
        sin,
        attention_mask,
        past_key_values=None,
        use_cache=False,
    ):
        hidden = inputs_embeds
        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.decoder_layers):
            past = past_key_values[i] if past_key_values is not None else None
            out = layer(
                hidden,
                cos,
                sin,
                attention_mask=attention_mask,
                past_key_value=past,
                use_cache=use_cache,
            )
            if use_cache:
                hidden, state = out
                new_cache.append(state)
            else:
                hidden = out
        hidden = self.norm(hidden)
        return (hidden, new_cache) if use_cache else hidden

    def _forward_features(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids_np = np.asarray(ops.convert_to_numpy(inputs["input_ids"])).astype(
            "int64"
        )
        batch, seq = input_ids_np.shape
        inputs_embeds = self.embed_tokens(ops.convert_to_tensor(input_ids_np))
        position_ids = self._positions(inputs.get("attention_mask"), batch, seq)
        cos, sin = rope_cos_sin(position_ids, self.rotary_dim, self.rope_theta)
        cos, sin = ops.convert_to_tensor(cos), ops.convert_to_tensor(sin)
        attn_mask = self._causal_mask(seq, seq, offset=0)
        return self._run_decoder(inputs_embeds, cos, sin, attn_mask)

    def call(self, inputs):
        """Return raw features. Use ``Qwen3_5Generate`` for logits / text."""
        return {"last_hidden_state": self._forward_features(inputs)}

    @classmethod
    def config_from_hf(cls, hf_config):
        c = hf_config.get("text_config", hf_config)
        rope = c.get("rope_parameters", c)
        return {
            "vocab_size": c["vocab_size"],
            "hidden_size": c["hidden_size"],
            "intermediate_size": c["intermediate_size"],
            "num_hidden_layers": c["num_hidden_layers"],
            "num_attention_heads": c["num_attention_heads"],
            "num_key_value_heads": c["num_key_value_heads"],
            "head_dim": c["head_dim"],
            "rms_norm_eps": c.get("rms_norm_eps", 1e-6),
            "rope_theta": rope.get("rope_theta", c.get("rope_theta", 10000000.0)),
            "partial_rotary_factor": rope.get("partial_rotary_factor", 0.25),
            "tie_word_embeddings": c.get("tie_word_embeddings", True),
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
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "rms_norm_eps": self.rms_norm_eps,
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
                "tie_word_embeddings": self.tie_word_embeddings,
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
    """Qwen3.5 with an LM head + greedy ``.generate()`` (text -> text)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = (
            None
            if self.tie_word_embeddings
            else layers.Dense(self.vocab_size, use_bias=False, name="lm_head")
        )

    def _lm_logits(self, hidden):
        if getattr(self, "lm_head", None) is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.embed_tokens.embeddings))

    def call(self, inputs):
        hidden = self._forward_features(inputs)
        return {"logits": self._lm_logits(hidden), "last_hidden_state": hidden}

    def generate(
        self, input_ids, attention_mask=None, max_new_tokens=128, eos_token_id=(248044,)
    ):
        """Greedy decoding with a hybrid KV / conv+recurrent cache."""
        input_ids_np = np.asarray(ops.convert_to_numpy(input_ids)).astype("int64")
        batch, prompt_len = input_ids_np.shape
        inputs_embeds = self.embed_tokens(ops.convert_to_tensor(input_ids_np))
        position_ids = self._positions(attention_mask, batch, prompt_len)
        cos, sin = rope_cos_sin(position_ids, self.rotary_dim, self.rope_theta)
        hidden, cache = self._run_decoder(
            inputs_embeds,
            ops.convert_to_tensor(cos),
            ops.convert_to_tensor(sin),
            self._causal_mask(prompt_len, prompt_len, offset=0),
            use_cache=True,
        )
        next_tok = np.asarray(
            ops.convert_to_numpy(
                ops.argmax(self._lm_logits(hidden[:, -1:, :]), axis=-1)
            )
        ).astype("int64")

        eos = {
            int(e)
            for e in (
                eos_token_id
                if isinstance(eos_token_id, (list, tuple))
                else [eos_token_id]
            )
        }
        first_eos = next(iter(eos)) if eos else 0
        finished = np.isin(next_tok[:, 0], list(eos))
        generated = [next_tok]
        cur_len = prompt_len
        for _ in range(max_new_tokens - 1):
            if finished.all():
                break
            pos = np.full((batch, 1), cur_len, dtype="int64")
            c, s = rope_cos_sin(pos, self.rotary_dim, self.rope_theta)
            step = self.embed_tokens(ops.convert_to_tensor(next_tok))
            hidden, cache = self._run_decoder(
                step,
                ops.convert_to_tensor(c),
                ops.convert_to_tensor(s),
                None,
                past_key_values=cache,
                use_cache=True,
            )
            next_tok = np.asarray(
                ops.convert_to_numpy(ops.argmax(self._lm_logits(hidden), axis=-1))
            ).astype("int64")
            next_tok[finished, 0] = first_eos
            generated.append(next_tok)
            cur_len += 1
            finished = finished | np.isin(next_tok[:, 0], list(eos))
        return np.concatenate(generated, axis=1)
