import keras
from keras import layers, ops

from kerasformers.base import SubclassedBaseModel
from kerasformers.base.constants import MASK_NEG

from .config import GPT2_CONFIG, GPT2_WEIGHTS
from .gpt2_layers import GPT2Block


@keras.saving.register_keras_serializable(package="kerasformers")
class GPT2Model(SubclassedBaseModel):
    """GPT-2 decoder-only transformer backbone (no LM head).

    Learned token (``wte``) + absolute-position (``wpe``) embeddings, a stack of
    pre-LayerNorm causal blocks, and a final LayerNorm (``ln_f``). Subclassed
    (imperative) model: the forward runs eagerly with ``keras.ops``. Returns raw
    features; use :class:`GPT2Generate` for logits / text.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: Feed-forward hidden width per block.
        num_layers: Number of decoder blocks.
        num_heads: Attention heads per block.
        max_position_embeddings: Size of the learned position table.
        norm_eps: LayerNorm epsilon.
        tie_embeddings: Whether :class:`GPT2Generate` ties the LM head to ``wte``.
    """

    HF_MODEL_TYPE = "gpt2"
    BASE_MODEL_CONFIG = GPT2_CONFIG
    BASE_WEIGHT_CONFIG = GPT2_WEIGHTS

    def __init__(
        self,
        vocab_size=50257,
        embed_dim=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        max_position_embeddings=1024,
        norm_eps=1e-5,
        tie_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.norm_eps = norm_eps
        self.tie_embeddings = tie_embeddings

        self.token_embedding = layers.Embedding(vocab_size, embed_dim, name="wte")
        self.wpe = layers.Embedding(max_position_embeddings, embed_dim, name="wpe")
        self.blocks = [
            GPT2Block(embed_dim, mlp_dim, num_heads, norm_eps, name=f"block_{i}")
            for i in range(num_layers)
        ]
        self.ln_f = layers.LayerNormalization(epsilon=norm_eps, name="ln_f")

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        attention_mask = inputs.get("attention_mask")
        positions = ops.broadcast_to(ops.arange(seq), (batch, seq))
        hidden = self.token_embedding(input_ids) + self.wpe(positions)
        mask = self.causal_mask(seq, attention_mask)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=mask)
        return {"last_hidden_state": self.ln_f(hidden)}

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["n_embd"],
            "mlp_dim": hf_config.get("n_inner") or 4 * hf_config["n_embd"],
            "num_layers": hf_config["n_layer"],
            "num_heads": hf_config["n_head"],
            "max_position_embeddings": hf_config["n_positions"],
            "norm_eps": hf_config.get("layer_norm_epsilon", 1e-5),
            "tie_embeddings": hf_config.get("tie_word_embeddings", True),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gpt2_hf_to_keras import transfer_gpt2_weights

        transfer_gpt2_weights(keras_model, hf_state_dict)

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        # Subclassed model: build it (weights are created on first call) before
        # loading the released .weights.h5 / sharded .weights.json.
        entry = cls.BASE_WEIGHT_CONFIG.get(variant, {})
        url = entry.get("url") if isinstance(entry, dict) else entry
        if not (load_weights and url):
            return super().from_release(
                variant,
                load_weights=load_weights,
                skip_mismatch=skip_mismatch,
                **kwargs,
            )
        model = super().from_release(variant, load_weights=False, **kwargs)
        model({"input_ids": ops.zeros((1, 8), dtype="int32")})
        cls.load_weights_from_url(model, url, skip_mismatch)
        return model

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "max_position_embeddings": self.max_position_embeddings,
                "norm_eps": self.norm_eps,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GPT2Generate(GPT2Model):
    """GPT-2 backbone + a (tied) language-model head and greedy ``.generate()``.

    ``call`` returns ``logits`` ``(batch, seq, vocab_size)`` and
    ``last_hidden_state``; :meth:`generate` does greedy decoding with a KV cache.
    The LM head is the transposed token embedding (GPT-2 ties them). Constructor
    ``Args`` are inherited from :class:`GPT2Model`.
    """

    def project(self, hidden):
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def call(self, inputs):
        hidden = super().call(inputs)["last_hidden_state"]
        return {"logits": self.project(hidden), "last_hidden_state": hidden}

    def generate(
        self, input_ids, attention_mask=None, max_new_tokens=128, eos_token_id=(50256,)
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, prompt_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        positions = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        hidden = self.token_embedding(input_ids) + self.wpe(positions)
        mask = self.causal_mask(prompt_len, attention_mask)
        cache = []
        for block in self.blocks:
            hidden, kv = block(hidden, attention_mask=mask, use_cache=True)
            cache.append(kv)
        hidden = self.ln_f(hidden)[:, -1:, :]
        next_tok = ops.cast(ops.argmax(self.project(hidden), axis=-1), "int32")

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
            hidden = self.token_embedding(next_tok) + self.wpe(pos)
            new_cache = []
            for i, block in enumerate(self.blocks):
                hidden, kv = block(hidden, past_key_value=cache[i], use_cache=True)
                new_cache.append(kv)
            hidden = self.ln_f(hidden)
            cache = new_cache
            next_tok = ops.cast(ops.argmax(self.project(hidden), axis=-1), "int32")
            next_tok = ops.cast(
                ops.where(finished[:, None], first_eos, next_tok), "int32"
            )
            generated.append(next_tok)
            cur_len += 1
            for e in eos:
                finished = ops.logical_or(finished, next_tok[:, 0] == e)
        return ops.convert_to_numpy(ops.concatenate(generated, axis=1))
