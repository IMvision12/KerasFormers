import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel
from kerasformers.models.deepseek_v3.deepseek_v3_layers import (
    DeepseekV3DecoderLayer,
    DeepseekV3RMSNorm,
    yarn_get_mscale,
)
from kerasformers.models.deepseek_v3.deepseek_v3_model import DeepseekV3Model

from .config import KIMI_K25_CONFIG, KIMI_K25_WEIGHTS_URLS
from .kimi_k25_layers import KimiK25MultimodalProjection
from .kimi_k25_vision import KimiK25VisionModel

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class KimiK25Model(SubclassedBaseModel):
    """Kimi K2.5 / K2.6 / K2.7-Code: MoonViT + DeepSeek-V3 MoE decoder.

    The text tower *is* DeepSeek-V3 (MLA + aux-loss-free DeepSeekMoE), so its
    decoder layers are reused verbatim; Kimi only changes the geometry (384
    routed experts, ``n_group = topk_group = 1`` so group-limited routing
    degenerates to a plain top-k, ``first_k_dense = 1``, ``norm_eps = 1e-5``).
    Images and video share one vision path: MoonViT patch-embeds each
    ``(t, h, w)`` clip, temporally averages and 2x2 merges it, and the projector
    lifts each merged patch into the text width. Merged patches then replace the
    ``image_token_id`` / ``video_token_id`` placeholders in the prompt.
    ``video_token_id`` equals ``vocab_size``, so placeholders are zeroed before
    the embedding lookup. Returns raw features; use :class:`KimiK25Generate` for
    logits / text.

    Args:
        vocab_size / embed_dim / num_layers / num_heads: Text geometry.
        mlp_dim: Dense-layer SwiGLU width (``intermediate_size``).
        moe_mlp_dim: Per-expert width (``moe_intermediate_size``).
        num_experts / num_experts_per_tok / n_shared_experts: MoE shape.
        n_group / topk_group / norm_topk_prob / routed_scaling_factor: Routing.
        first_k_dense: Leading dense layers (1).
        q_lora_rank / kv_lora_rank: MLA bottlenecks.
        qk_nope_head_dim / qk_rope_head_dim / v_head_dim: MLA per-head splits.
        rope_theta / rope_scaling: Text rope (yarn).
        norm_eps: Text RMSNorm epsilon.
        max_position_embeddings: Used by the yarn attention-factor default.
        tie_embeddings: Whether :class:`KimiK25Generate` ties the LM head.
        vision_embed_dim / vision_depth / vision_num_heads / vision_mlp_dim /
        vision_patch_size: MoonViT geometry.
        pos_emb_height / pos_emb_width / pos_emb_time: Learned position tables.
        merge_kernel: Spatial patch-merge kernel (2, 2).
        vision_rope_theta: MoonViT 2D-rope base.
        projection_hidden_size: Projector ``pre_norm`` width (= vision width).
        projection_norm_eps: Projector ``pre_norm`` epsilon.
        image_token_id / video_token_id: Placeholder ids replaced by patches.
        vision_start_token_id / vision_end_token_id: Media delimiters, carried
            for the processor's benefit.
    """

    HF_MODEL_TYPE = "kimi_k25"
    BASE_MODEL_CONFIG = KIMI_K25_CONFIG
    BASE_WEIGHT_CONFIG = KIMI_K25_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=163840,
        embed_dim=7168,
        num_layers=61,
        num_heads=64,
        mlp_dim=18432,
        moe_mlp_dim=2048,
        num_experts=384,
        num_experts_per_tok=8,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.827,
        first_k_dense=1,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_theta=50000.0,
        rope_scaling=None,
        norm_eps=1e-5,
        max_position_embeddings=262144,
        tie_embeddings=False,
        vision_embed_dim=1152,
        vision_depth=27,
        vision_num_heads=16,
        vision_mlp_dim=4304,
        vision_patch_size=14,
        pos_emb_height=64,
        pos_emb_width=64,
        pos_emb_time=4,
        merge_kernel=(2, 2),
        vision_rope_theta=10000.0,
        projection_hidden_size=1152,
        projection_norm_eps=1e-5,
        image_token_id=163605,
        video_token_id=163840,
        vision_start_token_id=163602,
        vision_end_token_id=163604,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.moe_mlp_dim = moe_mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense = first_k_dense
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = dict(rope_scaling) if rope_scaling else None
        self.norm_eps = norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.tie_embeddings = tie_embeddings
        self.vision_embed_dim = vision_embed_dim
        self.vision_depth = vision_depth
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_patch_size = vision_patch_size
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.pos_emb_time = pos_emb_time
        self.merge_kernel = tuple(merge_kernel)
        self.vision_rope_theta = vision_rope_theta
        self.projection_hidden_size = projection_hidden_size
        self.projection_norm_eps = projection_norm_eps
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.softmax_scale = self.qk_head_dim**-0.5
        # Like DeepSeek-V3, the yarn mscale^2 correction is folded into the scale.
        scaling_cfg = self.rope_scaling or {}
        rope_type = scaling_cfg.get("rope_type", scaling_cfg.get("type", "default"))
        if rope_type != "default":
            mscale_all_dim = scaling_cfg.get("mscale_all_dim", 0)
            factor = scaling_cfg.get("factor")
            if factor is None and scaling_cfg.get("original_max_position_embeddings"):
                factor = (
                    max_position_embeddings
                    / scaling_cfg["original_max_position_embeddings"]
                )
            if mscale_all_dim and factor:
                mscale = yarn_get_mscale(factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inv_freq, self.attention_scaling = DeepseekV3Model.build_rope(
            qk_rope_head_dim, rope_theta, self.rope_scaling, max_position_embeddings
        )

        self.vision_tower = KimiK25VisionModel(
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            mlp_dim=vision_mlp_dim,
            patch_size=vision_patch_size,
            pos_emb_height=pos_emb_height,
            pos_emb_width=pos_emb_width,
            pos_emb_time=pos_emb_time,
            merge_kernel=merge_kernel,
            rope_theta=vision_rope_theta,
            name="vision_tower",
        )
        self.mm_projector = KimiK25MultimodalProjection(
            vision_embed_dim * self.merge_kernel[0] * self.merge_kernel[1],
            embed_dim,
            norm_eps=projection_norm_eps,
            name="mm_projector",
        )
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            DeepseekV3DecoderLayer(
                embed_dim,
                num_heads,
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                self.softmax_scale,
                use_moe=i >= first_k_dense,
                mlp_dim=mlp_dim,
                moe_mlp_dim=moe_mlp_dim,
                shared_mlp_dim=moe_mlp_dim * n_shared_experts,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_group=n_group,
                topk_group=topk_group,
                norm_topk_prob=norm_topk_prob,
                routed_scaling_factor=routed_scaling_factor,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = DeepseekV3RMSNorm(eps=norm_eps, name="final_norm")

    def build_for_transfer(self):
        pixel_values = ops.zeros(
            (4, 3, self.vision_patch_size, self.vision_patch_size), dtype="float32"
        )
        grid = ops.convert_to_tensor([[1, 2, 2]], dtype="int32")
        self(
            {
                "input_ids": ops.convert_to_tensor([[self.image_token_id, 0, 0, 0]]),
                "pixel_values": pixel_values,
                "image_grid_thw": grid,
            }
        )

    def rope_tables(self, position_ids):
        inv_freq = ops.convert_to_tensor(self.inv_freq)
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        return (
            ops.cast(ops.cos(freqs) * self.attention_scaling, self.compute_dtype),
            ops.cast(ops.sin(freqs) * self.attention_scaling, self.compute_dtype),
        )

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def get_image_features(self, pixel_values, grid_thw):
        """Patches -> merged tokens -> text width, ``(num_merged, embed_dim)``."""
        return self.mm_projector(self.vision_tower(pixel_values, grid_thw))

    def merge_media(self, hidden, input_ids, features, token_id):
        """Scatter ``features`` into ``hidden`` wherever ``input_ids == token_id``.

        The placeholders are consumed in row-major order, so a running count of
        the mask gives each one its feature row -- a gather, which every backend
        differentiates and compiles, unlike a scatter.
        """
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        flat_ids = ops.reshape(input_ids, (batch * seq,))
        flat_hidden = ops.reshape(hidden, (batch * seq, self.embed_dim))
        mask = ops.equal(flat_ids, token_id)
        idx = ops.cumsum(ops.cast(mask, "int32")) - 1
        gathered = ops.take(features, ops.maximum(idx, 0), axis=0)
        flat_hidden = ops.where(
            mask[:, None], ops.cast(gathered, flat_hidden.dtype), flat_hidden
        )
        return ops.reshape(flat_hidden, (batch, seq, self.embed_dim))

    def embed_inputs(
        self,
        input_ids,
        pixel_values,
        image_grid_thw,
        pixel_values_videos,
        video_grid_thw,
    ):
        # video_token_id == vocab_size, i.e. out of the embedding range, so both
        # placeholder kinds are zeroed before the lookup and overwritten after.
        media = ops.logical_or(
            ops.equal(input_ids, self.image_token_id),
            ops.equal(input_ids, self.video_token_id),
        )
        hidden = self.token_embedding(ops.where(media, 0, input_ids))
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            hidden = self.merge_media(
                hidden, input_ids, image_embeds, self.image_token_id
            )
        if pixel_values_videos is not None:
            video_embeds = self.get_image_features(pixel_values_videos, video_grid_thw)
            hidden = self.merge_media(
                hidden, input_ids, video_embeds, self.video_token_id
            )
        return hidden

    def forward_features(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        hidden = self.embed_inputs(
            input_ids,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_values_videos"),
            inputs.get("video_grid_thw"),
        )
        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables(position_ids)
        attn_mask = self.causal_mask(seq, inputs.get("attention_mask"))
        for layer in self.decoder_layers:
            hidden = layer(hidden, cos, sin, attention_mask=attn_mask)
        return self.final_norm(hidden)

    def call(self, inputs):
        return {"last_hidden_state": self.forward_features(inputs)}

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config.get("text_config") or {}
        vision = hf_config.get("vision_config") or {}

        def vision_get(default, *names):
            for name in names:
                if name in vision:
                    return vision[name]
            return default

        config = DeepseekV3Model.config_from_hf(text)
        merge = tuple(vision_get((2, 2), "merge_kernel_size"))
        vision_rope = vision.get("rope_parameters") or {}
        config.update(
            {
                "tie_embeddings": bool(
                    hf_config.get(
                        "tie_word_embeddings", text.get("tie_word_embeddings")
                    )
                    or False
                ),
                "vision_embed_dim": vision_get(1152, "hidden_size", "vt_hidden_size"),
                "vision_depth": vision_get(
                    27, "num_hidden_layers", "vt_num_hidden_layers"
                ),
                "vision_num_heads": vision_get(
                    16, "num_attention_heads", "vt_num_attention_heads"
                ),
                "vision_mlp_dim": vision_get(
                    4304, "intermediate_size", "vt_intermediate_size"
                ),
                "vision_patch_size": vision_get(14, "patch_size"),
                "pos_emb_height": vision_get(
                    64, "pos_emb_height", "init_pos_emb_height"
                ),
                "pos_emb_width": vision_get(64, "pos_emb_width", "init_pos_emb_width"),
                "pos_emb_time": vision_get(4, "pos_emb_time", "init_pos_emb_time"),
                "merge_kernel": merge,
                "vision_rope_theta": vision_rope.get("rope_theta", 10000.0),
                "projection_hidden_size": hf_config.get("projection_hidden_size", 1152),
                "projection_norm_eps": hf_config.get("projection_layer_norm_eps")
                or vision.get("projector_ln_eps")
                or 1e-5,
                "image_token_id": hf_config.get(
                    "image_token_id",
                    hf_config.get("media_placeholder_token_id", 163605),
                ),
                "video_token_id": hf_config.get("video_token_id", 163840),
                "vision_start_token_id": hf_config.get("vision_start_token_id", 163602),
                "vision_end_token_id": hf_config.get("vision_end_token_id", 163604),
            }
        )
        return config

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_kimi_k25_hf_to_keras import transfer_kimi_k25_weights

        transfer_kimi_k25_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "moe_mlp_dim": self.moe_mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "n_shared_experts": self.n_shared_experts,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "norm_topk_prob": self.norm_topk_prob,
                "routed_scaling_factor": self.routed_scaling_factor,
                "first_k_dense": self.first_k_dense,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "rope_theta": self.rope_theta,
                "rope_scaling": self.rope_scaling,
                "norm_eps": self.norm_eps,
                "max_position_embeddings": self.max_position_embeddings,
                "tie_embeddings": self.tie_embeddings,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_depth": self.vision_depth,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_patch_size": self.vision_patch_size,
                "pos_emb_height": self.pos_emb_height,
                "pos_emb_width": self.pos_emb_width,
                "pos_emb_time": self.pos_emb_time,
                "merge_kernel": self.merge_kernel,
                "vision_rope_theta": self.vision_rope_theta,
                "projection_hidden_size": self.projection_hidden_size,
                "projection_norm_eps": self.projection_norm_eps,
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "vision_start_token_id": self.vision_start_token_id,
                "vision_end_token_id": self.vision_end_token_id,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class KimiK25Generate(KimiK25Model, BaseGeneration):
    """Kimi K2.5 with an LM head + fast ``.generate()`` (image/video+text -> text).

    Media only enters through the prefill, so ``pixel_values`` / ``image_grid_thw``
    (and the video pair) are passed to ``generate`` as prefill kwargs; decode
    steps run text-only against the MLA cache. As in DeepSeek-V3 the cache stores
    expanded per-head keys and values as a per-layer ``(k, v)`` tuple, since their
    head dims differ (k: nope+rope = 192, v: ``v_head_dim`` = 128).
    """

    # <|im_end|> (163586), per the published generation_config.
    eos_token_id = (163586,)

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
        hidden = self.forward_features(inputs)
        return {"logits": self.project(hidden), "last_hidden_state": hidden}

    def build_cache(
        self,
        token_ids,
        padding_mask,
        max_len,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        batch = int(token_ids.shape[0])
        seq = int(token_ids.shape[1])
        hidden = self.embed_inputs(
            ops.cast(token_ids, "int32"),
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
        )
        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables(position_ids)
        causal = self.causal_mask(seq, padding_mask)
        caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden, cos, sin, attention_mask=causal, use_cache=True
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
            caches.append((ck, cv))
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return tuple(caches), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        batch = int(token_ids.shape[0])
        max_len = int(cache[0][0].shape[2])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        new_cache = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[i][0], cache[i][1], pos, key_mask
            )
            new_cache.append((ck, cv))
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, tuple(new_cache)
