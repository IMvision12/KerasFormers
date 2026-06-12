import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .config import MISTRAL4_CONFIG, MISTRAL4_WEIGHTS_URLS
from .mistral4_layers import Mistral4DecoderLayer, Mistral4RMSNorm

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Mistral4Model(SubclassedBaseModel):
    """Mistral Large 3 text decoder backbone (no LM head).

    ``token_embedding -> num_layers x Mistral4DecoderLayer -> final RMSNorm``
    with multi-head latent attention (low-rank q/kv compression, split
    non-rotary / interleaved-rotary head dims, Llama-4 position-temperature
    query scaling) and a DeepSeek-V3-style mixture of experts (128 routed
    experts top-4 + one shared expert) on every layer after the first
    ``first_k_dense_replace`` dense ones. This port evaluates all experts
    densely and combines by the routing weights. Note the official
    ``mistralai/Mistral-Large-3-*`` repos ship mistral-native consolidated
    shards (no HF-format weights), so the variants here are
    architecture-only: build with ``load_weights=False`` or load an HF-format
    export via ``from_weights("hf:...")``. Returns raw features; use
    :class:`Mistral4Generate` for logits / text.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model / residual-stream width.
        mlp_dim_dense: Dense layers' SwiGLU hidden width.
        moe_mlp_dim: Per-routed-expert hidden width.
        num_layers: Number of decoder blocks.
        num_heads: Attention heads.
        num_experts: Routed expert count.
        num_experts_per_tok: Top-k experts per token.
        n_shared_experts: Shared-expert width multiplier.
        n_group / topk_group: Expert-group routing parameters.
        norm_topk_prob: Renormalize kept top-k weights.
        routed_scaling_factor: Final routing-weight scale.
        first_k_dense_replace: Layers before this index use a dense SwiGLU.
        q_lora_rank / kv_lora_rank: MLA compression ranks.
        qk_nope_head_dim / qk_rope_head_dim: Non-rotary / rotary head dims.
        v_head_dim: Per-head value dim.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        rope_interleave: Whether rotary dims are stored interleaved.
        llama4_beta: Attention-temperature beta (``None`` disables).
        llama4_original_max: Temperature position threshold.
        tie_embeddings: Whether :class:`Mistral4Generate` ties the LM head.
    """

    HF_MODEL_TYPE = "mistral4"
    BASE_MODEL_CONFIG = MISTRAL4_CONFIG
    BASE_WEIGHT_CONFIG = MISTRAL4_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=131072,
        embed_dim=7168,
        mlp_dim_dense=16384,
        moe_mlp_dim=4096,
        num_layers=61,
        num_heads=128,
        num_experts=128,
        num_experts_per_tok=4,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        first_k_dense_replace=3,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        norm_eps=1e-6,
        rope_theta=10000.0,
        rope_interleave=True,
        llama4_beta=0.1,
        llama4_original_max=8192,
        tie_embeddings=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim_dense = mlp_dim_dense
        self.moe_mlp_dim = moe_mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.rope_interleave = rope_interleave
        self.llama4_beta = llama4_beta
        self.llama4_original_max = llama4_original_max
        self.tie_embeddings = tie_embeddings
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Mistral4DecoderLayer(
                embed_dim,
                mlp_dim_dense,
                moe_mlp_dim,
                num_heads,
                is_moe=i >= first_k_dense_replace,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_shared_experts=n_shared_experts,
                n_group=n_group,
                topk_group=topk_group,
                norm_topk_prob=norm_topk_prob,
                routed_scaling_factor=routed_scaling_factor,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                rope_interleave=rope_interleave,
                llama4_beta=llama4_beta,
                llama4_original_max=llama4_original_max,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Mistral4RMSNorm(eps=norm_eps, name="final_norm")

    def rope_tables(self, position_ids):
        hd = self.qk_rope_head_dim
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
        hidden = self.token_embedding(input_ids)
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
            hidden = layer(hidden, cos, sin, position_ids, attention_mask=attn_mask)
        return {"last_hidden_state": self.final_norm(hidden)}

    @classmethod
    def config_from_hf(cls, hf_config):
        rope_params = hf_config.get("rope_parameters") or {}
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "mlp_dim_dense": hf_config["intermediate_size"],
            "moe_mlp_dim": hf_config["moe_intermediate_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_experts": hf_config["n_routed_experts"],
            "num_experts_per_tok": hf_config.get("num_experts_per_tok", 4),
            "n_shared_experts": hf_config.get("n_shared_experts", 1),
            "n_group": hf_config.get("n_group") or 1,
            "topk_group": hf_config.get("topk_group") or 1,
            "norm_topk_prob": bool(hf_config.get("norm_topk_prob", True)),
            "routed_scaling_factor": hf_config.get("routed_scaling_factor", 1.0),
            "first_k_dense_replace": hf_config.get("first_k_dense_replace", 0),
            "q_lora_rank": hf_config.get("q_lora_rank"),
            "kv_lora_rank": hf_config["kv_lora_rank"],
            "qk_nope_head_dim": hf_config["qk_nope_head_dim"],
            "qk_rope_head_dim": hf_config["qk_rope_head_dim"],
            "v_head_dim": hf_config.get("v_head_dim", 128),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": rope_params.get(
                "rope_theta", hf_config.get("rope_theta", 10000.0)
            ),
            "rope_interleave": bool(hf_config.get("rope_interleave", True)),
            "llama4_beta": rope_params.get("llama_4_scaling_beta"),
            "llama4_original_max": rope_params.get(
                "original_max_position_embeddings", 8192
            ),
            "tie_embeddings": hf_config.get("tie_word_embeddings", False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_mistral4_hf_to_keras import transfer_mistral4_weights

        transfer_mistral4_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim_dense": self.mlp_dim_dense,
                "moe_mlp_dim": self.moe_mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "n_shared_experts": self.n_shared_experts,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "norm_topk_prob": self.norm_topk_prob,
                "routed_scaling_factor": self.routed_scaling_factor,
                "first_k_dense_replace": self.first_k_dense_replace,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "rope_interleave": self.rope_interleave,
                "llama4_beta": self.llama4_beta,
                "llama4_original_max": self.llama4_original_max,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Mistral4Generate(Mistral4Model, BaseGeneration):
    """Mistral Large 3 backbone + a language-model head and fast ``.generate()``.

    Adds a bias-free ``lm_head`` on top of :class:`Mistral4Model` (the
    checkpoints do not tie embeddings). ``call`` returns ``logits``
    ``(batch, seq, vocab_size)`` and ``last_hidden_state``. Fast generation
    comes from :class:`~kerasformers.base.BaseGeneration`, fulfilled here by
    ``build_cache`` / ``call_with_cache`` over a pair of fixed caches (keys
    use ``qk_head_dim`` slots, values ``v_head_dim``). Constructor ``Args``
    are inherited from :class:`Mistral4Model`.
    """

    # Mistral </s> stop id. Explicit generate() args override this.
    eos_token_id = (2,)

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
        # Parallel prefill into fixed key (qk_head_dim) and value (v_head_dim)
        # caches, (B, num_layers, num_heads, max_len, dim) each. Returns
        # ((k_cache, v_cache), last-token logits).
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
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
        hidden = self.token_embedding(token_ids)
        k_caches, v_caches = [], []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden,
                cos,
                sin,
                position_ids,
                attention_mask=causal,
                use_cache=True,
            )
            ck = ops.slice_update(
                ops.zeros(
                    (batch, self.num_heads, max_len, self.qk_head_dim), dtype=k.dtype
                ),
                (0, 0, 0, 0),
                k,
            )
            cv = ops.slice_update(
                ops.zeros(
                    (batch, self.num_heads, max_len, self.v_head_dim), dtype=v.dtype
                ),
                (0, 0, 0, 0),
                v,
            )
            k_caches.append(ck)
            v_caches.append(cv)
        cache = (ops.stack(k_caches, axis=1), ops.stack(v_caches, axis=1))
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return cache, logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        # One decode step at position ``cache_update_index``.
        k_cache, v_cache = cache
        batch = int(token_ids.shape[0])
        max_len = int(k_cache.shape[3])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        k_caches, v_caches = [], []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h,
                cos,
                sin,
                positions,
                k_cache[:, i],
                v_cache[:, i],
                pos,
                key_mask,
            )
            k_caches.append(ck)
            v_caches.append(cv)
        cache = (ops.stack(k_caches, axis=1), ops.stack(v_caches, axis=1))
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
