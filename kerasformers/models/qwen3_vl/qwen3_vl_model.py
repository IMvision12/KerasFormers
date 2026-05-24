import keras
from keras import layers, ops

from kerasformers.base import BaseModel
from kerasformers.models.qwen2_vl.qwen2_vl_model import (
    _MASK_NEG,
    Qwen2VLModel,
    _QwenVLGenerateMixin,
    vision_rotary_cos_sin,
)

from .config import QWEN3_VL_CONFIG, QWEN3_VL_TOKENS, QWEN3_VL_WEIGHTS
from .qwen3_vl_layers import (
    Qwen3VLRMSNorm,
    Qwen3VLTextDecoderLayer,
    Qwen3VLVisionBlock,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
)


def qwen3_text_cos_sin(position_ids, head_dim, theta, mrope_section):
    """Interleaved M-RoPE cos/sin (Qwen3-VL).

    Builds per-axis frequencies then interleaves them channel-wise — T on
    channels ``0,3,6,...``, H on ``1,4,...`` (up to ``mrope_section[1]*3``),
    W on ``2,5,...`` (up to ``mrope_section[2]*3``), the tail staying T — rather
    than the contiguous T/H/W sections of Qwen2.x. Returns merged
    ``(batch, seq, head_dim)`` cos/sin tensors.
    """
    inv_freq = 1.0 / ops.power(
        theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    sel = [0] * (head_dim // 2)
    for dim, offset in ((1, 1), (2, 2)):
        for c in range(offset, mrope_section[dim] * 3, 3):
            sel[c] = dim
    sel = ops.convert_to_tensor(sel, dtype="int32")
    freqs_t = (
        ops.where(sel == 0, freqs[0], 0.0)
        + ops.where(sel == 1, freqs[1], 0.0)
        + ops.where(sel == 2, freqs[2], 0.0)
    )
    emb = ops.concatenate([freqs_t, freqs_t], axis=-1)
    return ops.cos(emb), ops.sin(emb)


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLVisionModel(layers.Layer):
    """Qwen3-VL vision tower: learned pos-embeds, full-attention GELU blocks,
    a final merger plus DeepStack mergers feeding the LLM's early layers."""

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        intermediate_size,
        out_hidden_size,
        num_position_embeddings,
        deepstack_visual_indexes,
        hidden_act="gelu_pytorch_tanh",
        patch_size=16,
        spatial_merge_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = tuple(deepstack_visual_indexes)
        self.hidden_act = hidden_act
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.head_dim = embed_dim // num_heads
        self.merge_unit = spatial_merge_size * spatial_merge_size
        self.num_grid_per_side = int(round(num_position_embeddings**0.5))

        self.patch_embed = Qwen3VLVisionPatchEmbed(embed_dim, name="patch_embed")
        self.blocks = [
            Qwen3VLVisionBlock(
                embed_dim, num_heads, intermediate_size, name=f"blocks_{i}"
            )
            for i in range(depth)
        ]
        self.merger = Qwen3VLVisionPatchMerger(
            out_hidden_size,
            embed_dim,
            spatial_merge_size,
            use_postshuffle_norm=False,
            name="merger",
        )
        self.deepstack_mergers = [
            Qwen3VLVisionPatchMerger(
                out_hidden_size,
                embed_dim,
                spatial_merge_size,
                use_postshuffle_norm=True,
                name=f"deepstack_merger_{i}",
            )
            for i in range(len(self.deepstack_visual_indexes))
        ]

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(self.num_position_embeddings, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def _interp_pos_embed(self, grid_rows):
        npos = self.num_grid_per_side
        m = self.spatial_merge_size
        pieces = []
        for t, h, w in grid_rows:
            hi = ops.linspace(0.0, float(npos - 1), h)
            wi = ops.linspace(0.0, float(npos - 1), w)
            hf = ops.cast(hi, "int32")
            wf = ops.cast(wi, "int32")
            hc = ops.minimum(hf + 1, npos - 1)
            wc = ops.minimum(wf + 1, npos - 1)
            dh = (hi - ops.cast(hf, "float32"))[:, None]
            dw = (wi - ops.cast(wf, "float32"))[None, :]
            i00 = ops.reshape(hf[:, None] * npos + wf[None, :], (-1,))
            i01 = ops.reshape(hf[:, None] * npos + wc[None, :], (-1,))
            i10 = ops.reshape(hc[:, None] * npos + wf[None, :], (-1,))
            i11 = ops.reshape(hc[:, None] * npos + wc[None, :], (-1,))
            w00 = ops.reshape((1 - dh) * (1 - dw), (-1, 1))
            w01 = ops.reshape((1 - dh) * dw, (-1, 1))
            w10 = ops.reshape(dh * (1 - dw), (-1, 1))
            w11 = ops.reshape(dh * dw, (-1, 1))
            emb = (
                ops.take(self.pos_embed, i00, axis=0) * w00
                + ops.take(self.pos_embed, i01, axis=0) * w01
                + ops.take(self.pos_embed, i10, axis=0) * w10
                + ops.take(self.pos_embed, i11, axis=0) * w11
            )
            emb = ops.reshape(emb, (1, h // m, m, w // m, m, self.embed_dim))
            emb = ops.transpose(emb, (0, 1, 3, 2, 4, 5))
            emb = ops.reshape(emb, (h * w, self.embed_dim))
            if t > 1:
                emb = ops.concatenate([emb] * t, axis=0)
            pieces.append(emb)
        return ops.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]

    def _full_mask(self, grid_rows, seq):
        cu = [0]
        for t, h, w in grid_rows:
            for _ in range(t):
                cu.append(cu[-1] + h * w)
        if len(cu) <= 2:
            return None
        seg = [0] * seq
        for i in range(len(cu) - 1):
            for j in range(cu[i], cu[i + 1]):
                seg[j] = i
        seg = ops.convert_to_tensor(seg, dtype="int32")
        mask = ops.where(seg[:, None] == seg[None, :], 0.0, _MASK_NEG)
        return ops.cast(mask, "float32")[None, None]

    def call(self, pixel_values, grid_thw):
        grid_rows = [
            tuple(int(v) for v in row)
            for row in ops.convert_to_numpy(ops.convert_to_tensor(grid_thw))
        ]
        seq = sum(t * h * w for t, h, w in grid_rows)
        hidden = self.patch_embed(pixel_values)
        hidden = hidden + self._interp_pos_embed(grid_rows)

        cos, sin = vision_rotary_cos_sin(
            grid_thw, self.head_dim, self.spatial_merge_size
        )
        mask = self._full_mask(grid_rows, seq)

        deepstack = []
        for i, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin, attention_mask=mask)
            if i in self.deepstack_visual_indexes:
                j = self.deepstack_visual_indexes.index(i)
                deepstack.append(self.deepstack_mergers[j](hidden))
        merged = self.merger(hidden)
        return merged, deepstack

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "out_hidden_size": self.out_hidden_size,
                "num_position_embeddings": self.num_position_embeddings,
                "deepstack_visual_indexes": self.deepstack_visual_indexes,
                "hidden_act": self.hidden_act,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLTextModel(layers.Layer):
    """Qwen3 causal decoder with DeepStack visual-feature injection.

    Identical to the Qwen3 decoder except that, during prefill, the i-th
    DeepStack feature map (scattered to a full ``(batch, seq, hidden)`` tensor by
    the model) is added to the output of decoder layer ``i``.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        mlp_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
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
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Qwen3VLTextDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Qwen3VLRMSNorm(eps=norm_eps, name="final_norm")

    def call(
        self,
        inputs_embeds,
        cos,
        sin,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        deepstack_full=None,
    ):
        hidden = inputs_embeds
        new_cache = [] if use_cache else None
        n_ds = 0 if deepstack_full is None else len(deepstack_full)
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
            if i < n_ds:
                hidden = hidden + deepstack_full[i]
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
class Qwen3VLModel(Qwen2VLModel):
    """Qwen3-VL multimodal model (vision + Qwen3 decoder + DeepStack).

    Reuses :class:`Qwen2VLModel`'s fusion / M-RoPE-index / generation machinery,
    but overrides the rotary tables (interleaved M-RoPE via
    :func:`qwen3_text_cos_sin`) and the input prep (which also scatters the vision
    tower's DeepStack features into ``extra["deepstack_full"]`` for the text
    decoder's early layers).
    """

    HF_MODEL_TYPE = "qwen3_vl"
    BASE_MODEL_CONFIG = QWEN3_VL_CONFIG
    BASE_WEIGHT_CONFIG = QWEN3_VL_WEIGHTS

    def __init__(
        self,
        vocab_size=151936,
        embed_dim=2048,
        mlp_dim=6144,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        norm_eps=1e-6,
        rope_theta=5000000.0,
        mrope_section=(24, 20, 20),
        tie_embeddings=True,
        vision_depth=24,
        vision_hidden_size=1024,
        vision_intermediate_size=4096,
        vision_num_heads=16,
        vision_out_hidden_size=None,
        vision_hidden_act="gelu_pytorch_tanh",
        num_position_embeddings=2304,
        deepstack_visual_indexes=(5, 11, 17),
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        image_token_id=QWEN3_VL_TOKENS["image_token_id"],
        video_token_id=QWEN3_VL_TOKENS["video_token_id"],
        vision_start_token_id=QWEN3_VL_TOKENS["vision_start_token_id"],
        vision_end_token_id=QWEN3_VL_TOKENS["vision_end_token_id"],
        **kwargs,
    ):
        BaseModel.__init__(self, **kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.mrope_section = tuple(mrope_section)
        self.tie_embeddings = tie_embeddings
        self.vision_depth = vision_depth
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_heads = vision_num_heads
        self.vision_out_hidden_size = vision_out_hidden_size or embed_dim
        self.vision_hidden_act = vision_hidden_act
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = tuple(deepstack_visual_indexes)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        self.tokens_per_second = 1

        self.visual = Qwen3VLVisionModel(
            embed_dim=vision_hidden_size,
            depth=vision_depth,
            num_heads=vision_num_heads,
            intermediate_size=vision_intermediate_size,
            out_hidden_size=self.vision_out_hidden_size,
            num_position_embeddings=num_position_embeddings,
            deepstack_visual_indexes=deepstack_visual_indexes,
            hidden_act=vision_hidden_act,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            name="visual",
        )
        self.language_model = Qwen3VLTextModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            norm_eps=norm_eps,
            name="language_model",
        )
        self.lm_head = (
            None
            if tie_embeddings
            else layers.Dense(vocab_size, use_bias=False, name="lm_head")
        )

    def _merged_cos_sin(self, position_ids):
        return qwen3_text_cos_sin(
            position_ids, self.head_dim, self.rope_theta, self.mrope_section
        )

    def _prepare_inputs(
        self,
        input_ids,
        pixel_values,
        image_grid_thw,
        attention_mask,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        inputs_embeds = self.language_model.token_embedding(input_ids)
        rope_deltas = ops.zeros((batch,), dtype="int32")
        extra = {}
        has_image = pixel_values is not None and image_grid_thw is not None
        has_video = pixel_values_videos is not None and video_grid_thw is not None
        image_grid = video_grid = None
        if has_image or has_video:
            ids_flat = ops.convert_to_numpy(ops.reshape(input_ids, (-1,))).tolist()
            flat = ops.reshape(inputs_embeds, (batch * seq, self.embed_dim))
            deepstack_full = None
            if has_image:
                image_grid = ops.cast(ops.convert_to_tensor(image_grid_thw), "int32")
                image_embeds, ds = self.visual(pixel_values, image_grid)
                idx_t = ops.reshape(
                    ops.convert_to_tensor(
                        [j for j, v in enumerate(ids_flat) if v == self.image_token_id],
                        dtype="int32",
                    ),
                    (-1, 1),
                )
                flat = ops.scatter_update(
                    flat, idx_t, ops.cast(image_embeds, flat.dtype)
                )
                deepstack_full = self._deepstack_full(ds, idx_t, batch, seq)
            if has_video:
                video_grid = ops.cast(ops.convert_to_tensor(video_grid_thw), "int32")
                video_embeds, vds = self.visual(pixel_values_videos, video_grid)
                vidx_t = ops.reshape(
                    ops.convert_to_tensor(
                        [j for j, v in enumerate(ids_flat) if v == self.video_token_id],
                        dtype="int32",
                    ),
                    (-1, 1),
                )
                flat = ops.scatter_update(
                    flat, vidx_t, ops.cast(video_embeds, flat.dtype)
                )
                vds_full = self._deepstack_full(vds, vidx_t, batch, seq)
                deepstack_full = (
                    vds_full
                    if deepstack_full is None
                    else [a + b for a, b in zip(deepstack_full, vds_full)]
                )
            inputs_embeds = ops.reshape(flat, (batch, seq, self.embed_dim))
            if deepstack_full is not None:
                extra = {"deepstack_full": deepstack_full}
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid, video_grid, attention_mask=attention_mask
            )
        else:
            pos = ops.broadcast_to(ops.arange(seq), (batch, seq))
            position_ids = ops.broadcast_to(pos, (3, batch, seq))
        return inputs_embeds, position_ids, rope_deltas, extra

    def _deepstack_full(self, deepstack, idx_t, batch, seq):
        out = []
        for emb in deepstack:
            z = ops.zeros((batch * seq, self.embed_dim), dtype=emb.dtype)
            z = ops.scatter_update(z, idx_t, ops.cast(emb, z.dtype))
            out.append(ops.reshape(z, (batch, seq, self.embed_dim)))
        return out

    @classmethod
    def config_from_hf(cls, hf_config):
        tc = hf_config.get("text_config", hf_config)
        vc = hf_config.get("vision_config", {})
        rope_scaling = tc.get("rope_scaling") or hf_config.get("rope_scaling") or {}
        mrope = rope_scaling.get("mrope_section", [24, 20, 20])
        hidden = tc["hidden_size"]
        heads = tc["num_attention_heads"]
        return {
            "vocab_size": tc["vocab_size"],
            "embed_dim": hidden,
            "mlp_dim": tc["intermediate_size"],
            "num_layers": tc["num_hidden_layers"],
            "num_heads": heads,
            "num_kv_heads": tc["num_key_value_heads"],
            "head_dim": tc.get("head_dim", hidden // heads),
            "norm_eps": tc.get("rms_norm_eps", 1e-6),
            "rope_theta": tc.get("rope_theta", 5000000.0),
            "mrope_section": tuple(mrope),
            "tie_embeddings": hf_config.get(
                "tie_word_embeddings", tc.get("tie_word_embeddings", False)
            ),
            "vision_depth": vc.get("depth", 24),
            "vision_hidden_size": vc.get("hidden_size", 1024),
            "vision_intermediate_size": vc.get("intermediate_size", 4096),
            "vision_num_heads": vc.get("num_heads", 16),
            "vision_out_hidden_size": vc.get("out_hidden_size", hidden),
            "vision_hidden_act": vc.get("hidden_act", "gelu_pytorch_tanh"),
            "num_position_embeddings": vc.get("num_position_embeddings", 2304),
            "deepstack_visual_indexes": tuple(
                vc.get("deepstack_visual_indexes", (5, 11, 17))
            ),
            "patch_size": vc.get("patch_size", 16),
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
        from .convert_qwen3_vl_hf_to_keras import transfer_qwen3_vl_weights

        transfer_qwen3_vl_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super(Qwen2VLModel, self).get_config()
        for k in [
            "vocab_size",
            "embed_dim",
            "mlp_dim",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "norm_eps",
            "rope_theta",
            "mrope_section",
            "tie_embeddings",
            "vision_depth",
            "vision_hidden_size",
            "vision_intermediate_size",
            "vision_num_heads",
            "vision_out_hidden_size",
            "vision_hidden_act",
            "num_position_embeddings",
            "deepstack_visual_indexes",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "in_channels",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ]:
            config[k] = getattr(self, k)
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLGenerate(_QwenVLGenerateMixin, Qwen3VLModel):
    """Qwen3-VL with an LM head + greedy ``.generate()`` (image+text -> text)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_lm_head()
