import keras
from keras import layers, ops

from kerasformers.models.qwen2_vl.qwen2_vl_model import (
    _MASK_NEG,
    Qwen2VLModel,
    vision_rotary_cos_sin,
)

from .config import QWEN2_5_VL_CONFIG, QWEN2_5_VL_TOKENS, QWEN2_5_VL_WEIGHTS
from .qwen2_5_vl_layers import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLRMSNorm,
    Qwen2_5_VLVisionBlock,
)


def get_window_index(grid_thw, window_size, spatial_merge_size, patch_size):
    """Window partition over merged-patch groups (mirrors HF get_window_index).

    Returns ``(window_index, cu_window_seqlens)``: a permutation tensor of the
    ``seq // merge_unit`` merge-unit groups into window-contiguous order, and a
    list of cumulative per-window sequence lengths (in patch units).
    """
    m = spatial_merge_size
    merge_unit = m * m
    vit_window = window_size // m // patch_size
    grid_rows = [
        tuple(int(v) for v in row)
        for row in ops.convert_to_numpy(ops.convert_to_tensor(grid_thw))
    ]
    window_index = []
    cu_window_seqlens = [0]
    offset = 0
    for t, h, w in grid_rows:
        lh, lw = h // m, w // m
        pad_h = (vit_window - lh % vit_window) % vit_window
        pad_w = (vit_window - lw % vit_window) % vit_window
        nwh = (lh + pad_h) // vit_window
        nww = (lw + pad_w) // vit_window
        index = ops.reshape(ops.arange(t * lh * lw), (t, lh, lw))
        index = ops.pad(index, [(0, 0), (0, pad_h), (0, pad_w)], constant_values=-100)
        index = ops.reshape(index, (t, nwh, vit_window, nww, vit_window))
        index = ops.transpose(index, (0, 1, 3, 2, 4))
        index = ops.reshape(index, (t * nwh * nww, vit_window * vit_window))
        for win in ops.convert_to_numpy(index).tolist():
            vals = [v for v in win if v != -100]
            window_index.extend(x + offset for x in vals)
            cu_window_seqlens.append(cu_window_seqlens[-1] + len(vals) * merge_unit)
        offset += t * lh * lw
    cu = [cu_window_seqlens[0]]
    for v in cu_window_seqlens[1:]:
        if v != cu[-1]:
            cu.append(v)
    return ops.convert_to_tensor(window_index, dtype="int32"), cu


def _segment_mask(cu_seqlens, total):
    """Additive block-diagonal mask (1,1,total,total) from cumulative seqlens."""
    seg = [0] * total
    for i in range(len(cu_seqlens) - 1):
        for j in range(int(cu_seqlens[i]), int(cu_seqlens[i + 1])):
            seg[j] = i
    seg = ops.convert_to_tensor(seg, dtype="int32")
    mask = ops.where(seg[:, None] == seg[None, :], 0.0, _MASK_NEG)
    return ops.cast(mask, "float32")[None, None]


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5_VLVisionModel(layers.Layer):
    """Qwen2.5-VL vision tower: windowed RMSNorm/SwiGLU blocks + 2x2 merger."""

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        intermediate_size,
        out_hidden_size,
        window_size,
        fullatt_block_indexes,
        patch_size=14,
        spatial_merge_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.out_hidden_size = out_hidden_size
        self.window_size = window_size
        self.fullatt_block_indexes = tuple(fullatt_block_indexes)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.head_dim = embed_dim // num_heads
        self.merge_unit = spatial_merge_size * spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(embed_dim, name="patch_embed")
        self.blocks = [
            Qwen2_5_VLVisionBlock(
                embed_dim, num_heads, intermediate_size, name=f"blocks_{i}"
            )
            for i in range(depth)
        ]
        self.merger = Qwen2_5_VLPatchMerger(
            out_hidden_size,
            embed_dim,
            spatial_merge_size,
            name="merger",
        )

    def call(self, pixel_values, grid_thw):
        grid_rows = [
            tuple(int(v) for v in row)
            for row in ops.convert_to_numpy(ops.convert_to_tensor(grid_thw))
        ]
        u = self.merge_unit
        seq = sum(t * h * w for t, h, w in grid_rows)

        hidden = self.patch_embed(pixel_values)

        cos, sin = vision_rotary_cos_sin(
            grid_thw, self.head_dim, self.spatial_merge_size
        )
        window_index, cu_window = get_window_index(
            grid_thw, self.window_size, self.spatial_merge_size, self.patch_size
        )

        hidden = ops.reshape(hidden, (seq // u, u, self.embed_dim))
        hidden = ops.reshape(
            ops.take(hidden, window_index, axis=0), (seq, self.embed_dim)
        )
        cos = ops.reshape(
            ops.take(
                ops.reshape(cos, (seq // u, u, self.head_dim)), window_index, axis=0
            ),
            (seq, -1),
        )
        sin = ops.reshape(
            ops.take(
                ops.reshape(sin, (seq // u, u, self.head_dim)), window_index, axis=0
            ),
            (seq, -1),
        )

        cu_full = [0]
        for t, h, w in grid_rows:
            for _ in range(t):
                cu_full.append(cu_full[-1] + h * w)
        full_mask = None if len(cu_full) <= 2 else _segment_mask(cu_full, seq)
        window_mask = _segment_mask(cu_window, seq)

        for i, block in enumerate(self.blocks):
            mask = full_mask if i in self.fullatt_block_indexes else window_mask
            hidden = block(hidden, cos, sin, attention_mask=mask)

        merged = self.merger(hidden)
        reverse = ops.argsort(window_index)
        return ops.take(merged, reverse, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "out_hidden_size": self.out_hidden_size,
                "window_size": self.window_size,
                "fullatt_block_indexes": self.fullatt_block_indexes,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5_VLTextModel(layers.Layer):
    """Qwen2.5 causal decoder (same as Qwen2): embed -> N blocks -> RMSNorm."""

    def __init__(
        self,
        vocab_size,
        embed_dim,
        mlp_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim=None,
        norm_eps=1e-6,
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
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Qwen2_5_VLDecoderLayer(
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
        self.final_norm = Qwen2_5_VLRMSNorm(eps=norm_eps, name="final_norm")

    def call(
        self,
        inputs_embeds,
        cos,
        sin,
        attention_mask=None,
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
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5_VLModel(Qwen2VLModel):
    """Qwen2.5-VL: Qwen2-VL text/fusion/generation with a windowed vision tower."""

    HF_MODEL_TYPE = "qwen2_5_vl"
    BASE_MODEL_CONFIG = QWEN2_5_VL_CONFIG
    BASE_WEIGHT_CONFIG = QWEN2_5_VL_WEIGHTS

    def __init__(
        self,
        vocab_size=151936,
        embed_dim=2048,
        mlp_dim=11008,
        num_layers=36,
        num_heads=16,
        num_kv_heads=2,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        mrope_section=(16, 24, 24),
        tie_embeddings=True,
        vision_depth=32,
        vision_hidden_size=1280,
        vision_intermediate_size=3420,
        vision_num_heads=16,
        vision_out_hidden_size=None,
        window_size=112,
        fullatt_block_indexes=(7, 15, 23, 31),
        tokens_per_second=2,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        image_token_id=QWEN2_5_VL_TOKENS["image_token_id"],
        video_token_id=QWEN2_5_VL_TOKENS["video_token_id"],
        vision_start_token_id=QWEN2_5_VL_TOKENS["vision_start_token_id"],
        vision_end_token_id=QWEN2_5_VL_TOKENS["vision_end_token_id"],
        **kwargs,
    ):
        from kerasformers.base import BaseModel

        BaseModel.__init__(self, **kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.mrope_section = tuple(mrope_section)
        self.tie_embeddings = tie_embeddings
        self.vision_depth = vision_depth
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_heads = vision_num_heads
        self.vision_out_hidden_size = vision_out_hidden_size or embed_dim
        self.window_size = window_size
        self.fullatt_block_indexes = tuple(fullatt_block_indexes)
        self.tokens_per_second = tokens_per_second
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.patch_dim = in_channels * temporal_patch_size * patch_size * patch_size

        self.visual = Qwen2_5_VLVisionModel(
            embed_dim=vision_hidden_size,
            depth=vision_depth,
            num_heads=vision_num_heads,
            intermediate_size=vision_intermediate_size,
            out_hidden_size=self.vision_out_hidden_size,
            window_size=window_size,
            fullatt_block_indexes=fullatt_block_indexes,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            name="visual",
        )
        self.language_model = Qwen2_5_VLTextModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=self.head_dim,
            norm_eps=norm_eps,
            name="language_model",
        )

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config.get("vision_config", {})
        rope_scaling = hf_config.get("rope_scaling") or {}
        mrope = rope_scaling.get("mrope_section", [16, 24, 24])
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "mlp_dim": hf_config["intermediate_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_kv_heads": hf_config["num_key_value_heads"],
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            "mrope_section": tuple(mrope),
            "tie_embeddings": hf_config.get("tie_word_embeddings", False),
            "vision_depth": vc.get("depth", 32),
            "vision_hidden_size": vc.get("hidden_size", 1280),
            "vision_intermediate_size": vc.get("intermediate_size", 3420),
            "vision_num_heads": vc.get("num_heads", 16),
            "vision_out_hidden_size": vc.get(
                "out_hidden_size", hf_config["hidden_size"]
            ),
            "window_size": vc.get("window_size", 112),
            "fullatt_block_indexes": tuple(
                vc.get("fullatt_block_indexes", (7, 15, 23, 31))
            ),
            "tokens_per_second": vc.get("tokens_per_second", 2),
            "patch_size": vc.get("patch_size", 14),
            "spatial_merge_size": vc.get("spatial_merge_size", 2),
            "temporal_patch_size": vc.get("temporal_patch_size", 2),
            "in_channels": vc.get("in_chans", vc.get("in_channels", 3)),
            "image_token_id": hf_config.get("image_token_id", 151655),
            "video_token_id": hf_config.get("video_token_id", 151656),
            "vision_start_token_id": hf_config.get("vision_start_token_id", 151652),
            "vision_end_token_id": hf_config.get("vision_end_token_id", 151653),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_qwen2_5_vl_hf_to_keras import transfer_qwen2_5_vl_weights

        transfer_qwen2_5_vl_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super(Qwen2VLModel, self).get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "mrope_section": self.mrope_section,
                "tie_embeddings": self.tie_embeddings,
                "vision_depth": self.vision_depth,
                "vision_hidden_size": self.vision_hidden_size,
                "vision_intermediate_size": self.vision_intermediate_size,
                "vision_num_heads": self.vision_num_heads,
                "vision_out_hidden_size": self.vision_out_hidden_size,
                "window_size": self.window_size,
                "fullatt_block_indexes": self.fullatt_block_indexes,
                "tokens_per_second": self.tokens_per_second,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.in_channels,
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "vision_start_token_id": self.vision_start_token_id,
                "vision_end_token_id": self.vision_end_token_id,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5_VLGenerate(Qwen2_5_VLModel):
    """Qwen2.5-VL with an LM head + greedy ``.generate()`` (image+text -> text).

    Adds a vocabulary projection on top of :class:`Qwen2_5_VLModel`: a separate
    bias-free ``lm_head`` when ``tie_embeddings`` is ``False``, otherwise the
    (transposed) token embedding (weight tying). ``call`` returns both ``logits``
    and ``last_hidden_state``; :meth:`generate` does greedy decoding with a KV
    cache and incremental M-RoPE (each new token's position is
    ``cache_len + rope_delta`` on all three axes). Image / video pixels are passed
    exactly as for :class:`Qwen2_5_VLModel`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = (
            None
            if self.tie_embeddings
            else layers.Dense(self.vocab_size, use_bias=False, name="lm_head")
        )

    def call(self, inputs):
        hidden = self._forward_features(inputs)
        logits = (
            self.lm_head(hidden)
            if self.lm_head is not None
            else ops.matmul(
                hidden, ops.transpose(self.language_model.token_embedding.embeddings)
            )
        )
        return {"logits": logits, "last_hidden_state": hidden}

    def generate(
        self,
        input_ids,
        pixel_values=None,
        image_grid_thw=None,
        attention_mask=None,
        max_new_tokens=128,
        eos_token_id=(151645,),
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, prompt_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        inputs_embeds, position_ids, rope_deltas, extra = self._prepare_inputs(
            input_ids,
            pixel_values,
            image_grid_thw,
            attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        cos, sin = self._merged_cos_sin(position_ids)
        hidden, cache = self.language_model(
            inputs_embeds,
            cos,
            sin,
            attention_mask=self._causal_mask(prompt_len, prompt_len, offset=0),
            use_cache=True,
            **extra,
        )
        emb = self.language_model.token_embedding.embeddings
        last = hidden[:, -1:, :]
        logits = (
            self.lm_head(last)
            if self.lm_head is not None
            else ops.matmul(last, ops.transpose(emb))
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
            pos = ops.broadcast_to(
                ops.reshape(cur_len + rope_deltas, (1, batch, 1)), (3, batch, 1)
            )
            step_cos, step_sin = self._merged_cos_sin(pos)
            step_embeds = self.language_model.token_embedding(next_tok)
            hidden, cache = self.language_model(
                step_embeds,
                step_cos,
                step_sin,
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
            )
            logits = (
                self.lm_head(hidden)
                if self.lm_head is not None
                else ops.matmul(hidden, ops.transpose(emb))
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
