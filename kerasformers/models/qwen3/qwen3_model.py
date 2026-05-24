import keras
import numpy as np
from keras import layers, ops

from kerasformers.base import BaseModel

from .config import QWEN3_CONFIG, QWEN3_WEIGHTS
from .qwen3_layers import Qwen3DecoderLayer, Qwen3RMSNorm, rope_cos_sin

_MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3Model(BaseModel):
    """Qwen3 decoder: token_embedding -> N QK-norm decoder layers -> RMSNorm."""

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
                hidden, kv = out
                new_cache.append(kv)
            else:
                hidden = out
        hidden = self.final_norm(hidden)
        return (hidden, new_cache) if use_cache else hidden

    def _forward_features(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids_np = np.asarray(ops.convert_to_numpy(inputs["input_ids"])).astype(
            "int64"
        )
        batch, seq = input_ids_np.shape
        inputs_embeds = self.token_embedding(ops.convert_to_tensor(input_ids_np))
        position_ids = self._positions(inputs.get("attention_mask"), batch, seq)
        cos, sin = rope_cos_sin(position_ids, self.head_dim, self.rope_theta)
        cos, sin = ops.convert_to_tensor(cos), ops.convert_to_tensor(sin)
        attn_mask = self._causal_mask(seq, seq, offset=0)
        return self._run_decoder(inputs_embeds, cos, sin, attn_mask)

    def call(self, inputs):
        """Return raw features. Use ``Qwen3Generate`` for logits / text."""
        return {"last_hidden_state": self._forward_features(inputs)}

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
    """Qwen3 with an LM head + greedy ``.generate()`` (text -> text)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = (
            None
            if self.tie_embeddings
            else layers.Dense(self.vocab_size, use_bias=False, name="lm_head")
        )

    def _lm_logits(self, hidden):
        if getattr(self, "lm_head", None) is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def call(self, inputs):
        hidden = self._forward_features(inputs)
        return {"logits": self._lm_logits(hidden), "last_hidden_state": hidden}

    def generate(
        self, input_ids, attention_mask=None, max_new_tokens=128, eos_token_id=(151645,)
    ):
        """Greedy decoding with a KV cache. Returns ``(batch, num_new)`` ids."""
        input_ids_np = np.asarray(ops.convert_to_numpy(input_ids)).astype("int64")
        batch, prompt_len = input_ids_np.shape
        inputs_embeds = self.token_embedding(ops.convert_to_tensor(input_ids_np))
        position_ids = self._positions(attention_mask, batch, prompt_len)
        cos, sin = rope_cos_sin(position_ids, self.head_dim, self.rope_theta)
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
            c, s = rope_cos_sin(pos, self.head_dim, self.rope_theta)
            step = self.token_embedding(ops.convert_to_tensor(next_tok))
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
