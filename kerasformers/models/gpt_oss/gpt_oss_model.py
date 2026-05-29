import math

import keras
from keras import layers, ops

from kerasformers.base import SubclassedBaseModel

from .config import GPT_OSS_CONFIG, GPT_OSS_WEIGHTS
from .gpt_oss_layers import GptOssDecoderLayer, GptOssRMSNorm

_MASK_NEG = -1e9


def yarn_inv_freq(head_dim, base, factor, beta_fast, beta_slow, orig_max, truncate):
    """YaRN-scaled inverse frequencies + cos/sin scaling (port of HF's YaRN init).

    Returns ``(inv_freq, attention_scaling)`` where ``inv_freq`` is a
    ``(head_dim // 2,)`` tensor and ``attention_scaling`` (mscale) multiplies the
    rotary cos/sin. All inputs are config constants, so this is recomputed cheaply
    each forward (no stored state).
    """
    dim = head_dim
    pos_freqs = ops.power(
        float(base), ops.arange(0, dim, 2, dtype="float32") / dim
    )  # (dim/2,)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)

    def correction_dim(num_rotations):
        return (dim * math.log(orig_max / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = correction_dim(beta_fast)
    high = correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0.0)
    high = min(high, dim - 1.0)
    if low == high:
        high += 0.001

    ramp = ops.clip(
        (ops.arange(dim // 2, dtype="float32") - low) / (high - low), 0.0, 1.0
    )
    extrapolation_factor = 1.0 - ramp
    inv_freq = (
        inv_interp * (1.0 - extrapolation_factor) + inv_extrap * extrapolation_factor
    )
    attention_scaling = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
    return inv_freq, attention_scaling


@keras.saving.register_keras_serializable(package="kerasformers")
class GptOssModel(SubclassedBaseModel):
    """GPT-OSS mixture-of-experts decoder-only transformer backbone (no LM head).

    ``token_embedding -> num_layers x GptOssDecoderLayer -> final RMSNorm``, with
    grouped-query attention + learned per-head attention sinks, alternating
    sliding-window / full causal attention, YaRN-scaled rotary positions, and a
    top-k sparse MoE feed-forward per layer. Subclassed (imperative) model: the
    forward runs eagerly with ``keras.ops``. Returns raw features; use
    :class:`GptOssGenerate` for logits / text.

    Args:
        vocab_size, embed_dim, mlp_dim, num_layers, num_heads,
        num_kv_heads, head_dim: standard decoder dimensions (``mlp_dim``
            is the *per-expert* hidden width).
        num_experts, num_experts_per_tok: MoE expert count and top-k.
        sliding_window: window size of the sliding-attention layers (even layers).
        norm_eps: RMSNorm epsilon.
        rope_theta, rope_factor, rope_beta_fast, rope_beta_slow, rope_truncate,
        rope_original_max_pos: YaRN rotary parameters.
        attention_bias: whether q/k/v/o carry a bias (GPT-OSS: True).
        tie_embeddings: whether :class:`GptOssGenerate` ties the LM head.
    """

    HF_MODEL_TYPE = "gpt_oss"
    BASE_MODEL_CONFIG = GPT_OSS_CONFIG
    BASE_WEIGHT_CONFIG = GPT_OSS_WEIGHTS

    def __init__(
        self,
        vocab_size=201088,
        embed_dim=2880,
        mlp_dim=2880,
        num_layers=24,
        num_heads=64,
        num_kv_heads=8,
        head_dim=64,
        num_experts=32,
        num_experts_per_tok=4,
        sliding_window=128,
        norm_eps=1e-5,
        rope_theta=150000.0,
        rope_factor=32.0,
        rope_beta_fast=32.0,
        rope_beta_slow=1.0,
        rope_truncate=False,
        rope_original_max_pos=4096,
        attention_bias=True,
        tie_embeddings=False,
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
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.sliding_window = sliding_window
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor
        self.rope_beta_fast = rope_beta_fast
        self.rope_beta_slow = rope_beta_slow
        self.rope_truncate = rope_truncate
        self.rope_original_max_pos = rope_original_max_pos
        self.attention_bias = attention_bias
        self.tie_embeddings = tie_embeddings

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            GptOssDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                num_experts,
                num_experts_per_tok,
                norm_eps,
                attention_bias,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = GptOssRMSNorm(eps=norm_eps, name="final_norm")

    def rope(self, position_ids):
        inv_freq, scaling = yarn_inv_freq(
            self.head_dim,
            self.rope_theta,
            self.rope_factor,
            self.rope_beta_fast,
            self.rope_beta_slow,
            self.rope_original_max_pos,
            self.rope_truncate,
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        return ops.cos(emb) * scaling, ops.sin(emb) * scaling

    def is_sliding(self, layer_idx):
        # HF: "sliding_attention" if (i + 1) % 2 else "full_attention" -> even i slides
        return bool((layer_idx + 1) % 2)

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
        cos, sin = self.rope(position_ids)

        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = ki <= qi
        full_mask = ops.cast(ops.where(causal, 0.0, _MASK_NEG), "float32")[None, None]
        sliding_keep = ops.logical_and(causal, ki > qi - self.sliding_window)
        sliding_mask = ops.cast(ops.where(sliding_keep, 0.0, _MASK_NEG), "float32")[
            None, None
        ]

        for i, layer in enumerate(self.decoder_layers):
            mask = sliding_mask if self.is_sliding(i) else full_mask
            hidden = layer(hidden, cos, sin, attention_mask=mask)
        return {"last_hidden_state": self.final_norm(hidden)}

    @classmethod
    def config_from_hf(cls, hf_config):
        rope = hf_config.get("rope_parameters") or hf_config.get("rope_scaling") or {}
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "mlp_dim": hf_config["intermediate_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_kv_heads": hf_config["num_key_value_heads"],
            "head_dim": hf_config.get("head_dim")
            or hf_config["hidden_size"] // hf_config["num_attention_heads"],
            "num_experts": hf_config["num_local_experts"],
            "num_experts_per_tok": hf_config["num_experts_per_tok"],
            "sliding_window": hf_config.get("sliding_window", 128),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-5),
            "rope_theta": rope.get("rope_theta", hf_config.get("rope_theta", 150000.0)),
            "rope_factor": rope.get("factor", 32.0),
            "rope_beta_fast": rope.get("beta_fast", 32.0),
            "rope_beta_slow": rope.get("beta_slow", 1.0),
            "rope_truncate": rope.get("truncate", False),
            "rope_original_max_pos": rope.get("original_max_position_embeddings", 4096),
            "attention_bias": hf_config.get("attention_bias", True),
            "tie_embeddings": hf_config.get("tie_word_embeddings", False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gpt_oss_hf_to_keras import transfer_gpt_oss_weights

        transfer_gpt_oss_weights(keras_model, hf_state_dict)

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
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "sliding_window": self.sliding_window,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "rope_factor": self.rope_factor,
                "rope_beta_fast": self.rope_beta_fast,
                "rope_beta_slow": self.rope_beta_slow,
                "rope_truncate": self.rope_truncate,
                "rope_original_max_pos": self.rope_original_max_pos,
                "attention_bias": self.attention_bias,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GptOssGenerate(GptOssModel):
    """GPT-OSS backbone + a language-model head and greedy ``.generate()``.

    Adds a bias-free ``lm_head`` (GPT-OSS does not tie embeddings). ``call``
    returns ``logits`` ``(batch, seq, vocab_size)`` and ``last_hidden_state``;
    :meth:`generate` does greedy decoding with a KV cache that respects the
    per-layer sliding window. Constructor ``Args`` are inherited from
    :class:`GptOssModel`.
    """

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

    def generate(
        self, input_ids, attention_mask=None, max_new_tokens=128, eos_token_id=(200002,)
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, prompt_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos, sin = self.rope(position_ids)

        qi = ops.arange(prompt_len)[:, None]
        ki = ops.arange(prompt_len)[None, :]
        causal = ki <= qi
        full_mask = ops.cast(ops.where(causal, 0.0, _MASK_NEG), "float32")[None, None]
        sliding_keep = ops.logical_and(causal, ki > qi - self.sliding_window)
        sliding_mask = ops.cast(ops.where(sliding_keep, 0.0, _MASK_NEG), "float32")[
            None, None
        ]

        hidden = self.token_embedding(input_ids)
        cache = []
        for i, layer in enumerate(self.decoder_layers):
            mask = sliding_mask if self.is_sliding(i) else full_mask
            hidden, kv = layer(hidden, cos, sin, attention_mask=mask, use_cache=True)
            cache.append(kv)
        hidden = self.final_norm(hidden)[:, -1:, :]
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
            c, s = self.rope(ops.full((batch, 1), cur_len, dtype="int32"))
            hidden = self.token_embedding(next_tok)
            cache_len = cur_len + 1
            kpos = ops.arange(cache_len)
            sliding_dec = ops.cast(
                ops.where(kpos > cur_len - self.sliding_window, 0.0, _MASK_NEG),
                "float32",
            )[None, None, None]
            new_cache = []
            for i, layer in enumerate(self.decoder_layers):
                mask = sliding_dec if self.is_sliding(i) else None
                hidden, kv = layer(
                    hidden,
                    c,
                    s,
                    attention_mask=mask,
                    past_key_value=cache[i],
                    use_cache=True,
                )
                new_cache.append(kv)
            hidden = self.final_norm(hidden)
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
