import keras
from keras import layers, ops

from kerasformers.base import SubclassedBaseModel

from .config import GPT_CONFIG, GPT_WEIGHTS
from .gpt_layers import GptBlock

_MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class GptModel(SubclassedBaseModel):
    """Original GPT (Radford et al. 2018) decoder-only transformer backbone.

    Learned token (``tokens_embed``) + absolute-position (``positions_embed``)
    embeddings followed by a stack of post-LayerNorm causal blocks. Unlike GPT-2
    there is no final LayerNorm. Subclassed (imperative) model whose forward runs
    eagerly with ``keras.ops``; use :class:`GptGenerate` for logits / text.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim: Feed-forward hidden width per block.
        num_layers: Number of decoder blocks.
        num_heads: Attention heads per block.
        max_position_embeddings: Size of the learned position table.
        norm_eps: LayerNorm epsilon.
        tie_embeddings: Whether :class:`GptGenerate` ties the LM head to
            ``tokens_embed``.
    """

    HF_MODEL_TYPE = "openai-gpt"
    BASE_MODEL_CONFIG = GPT_CONFIG
    BASE_WEIGHT_CONFIG = GPT_WEIGHTS

    def __init__(
        self,
        vocab_size=40478,
        embed_dim=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        max_position_embeddings=512,
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

        self.tokens_embed = layers.Embedding(vocab_size, embed_dim, name="tokens_embed")
        self.positions_embed = layers.Embedding(
            max_position_embeddings, embed_dim, name="positions_embed"
        )
        self.blocks = [
            GptBlock(embed_dim, mlp_dim, num_heads, norm_eps, name=f"block_{i}")
            for i in range(num_layers)
        ]

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, _MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * _MASK_NEG
        return mask

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        attention_mask = inputs.get("attention_mask")
        positions = ops.broadcast_to(ops.arange(seq), (batch, seq))
        hidden = self.tokens_embed(input_ids) + self.positions_embed(positions)
        mask = self.causal_mask(seq, attention_mask)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=mask)
        return {"last_hidden_state": hidden}

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
        from .convert_gpt_hf_to_keras import transfer_gpt_weights

        transfer_gpt_weights(keras_model, hf_state_dict)

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
class GptGenerate(GptModel):
    """GPT backbone + a (tied) language-model head and greedy ``.generate()``.

    ``call`` returns ``logits`` ``(batch, seq, vocab_size)`` and
    ``last_hidden_state``; :meth:`generate` does greedy decoding with a KV cache.
    The LM head is the transposed token embedding. Constructor ``Args`` are
    inherited from :class:`GptModel`.
    """

    def project(self, hidden):
        return ops.matmul(hidden, ops.transpose(self.tokens_embed.embeddings))

    def call(self, inputs):
        hidden = super().call(inputs)["last_hidden_state"]
        return {"logits": self.project(hidden), "last_hidden_state": hidden}

    def generate(
        self, input_ids, attention_mask=None, max_new_tokens=128, eos_token_id=()
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, prompt_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        positions = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        hidden = self.tokens_embed(input_ids) + self.positions_embed(positions)
        mask = self.causal_mask(prompt_len, attention_mask)
        cache = []
        for block in self.blocks:
            hidden, kv = block(hidden, attention_mask=mask, use_cache=True)
            cache.append(kv)
        next_tok = ops.cast(
            ops.argmax(self.project(hidden[:, -1:, :]), axis=-1), "int32"
        )

        eos = [int(e) for e in (eos_token_id or ())]
        first_eos = eos[0] if eos else 0
        finished = ops.zeros((batch,), dtype="bool")
        for e in eos:
            finished = ops.logical_or(finished, next_tok[:, 0] == e)
        generated = [next_tok]
        cur_len = prompt_len
        for _ in range(max_new_tokens - 1):
            if eos and bool(ops.all(finished)):
                break
            pos = ops.full((batch, 1), cur_len, dtype="int32")
            hidden = self.tokens_embed(next_tok) + self.positions_embed(pos)
            new_cache = []
            for i, block in enumerate(self.blocks):
                hidden, kv = block(hidden, past_key_value=cache[i], use_cache=True)
                new_cache.append(kv)
            cache = new_cache
            next_tok = ops.cast(ops.argmax(self.project(hidden), axis=-1), "int32")
            if eos:
                next_tok = ops.cast(
                    ops.where(finished[:, None], first_eos, next_tok), "int32"
                )
            generated.append(next_tok)
            cur_len += 1
            for e in eos:
                finished = ops.logical_or(finished, next_tok[:, 0] == e)
        return ops.convert_to_numpy(ops.concatenate(generated, axis=1))
