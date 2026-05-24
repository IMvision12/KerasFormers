"""Qwen2-VL — multimodal (image/video + text) causal LLM in pure Keras 3.

``Qwen2VLModel`` is a subclassed :class:`BaseModel` (not Functional): the
vision sequence length, the number of image placeholder tokens, and the decode
step count are all data dependent, so the forward pass is written imperatively
with ``keras.ops``. The vision tower and the Qwen2 text decoder are exposed as
``model.visual`` / ``model.language_model`` (mirroring HF) for the generation
loop.

Weights load on the fly from Hugging Face:

    model = Qwen2VLModel.from_weights("hf:Qwen/Qwen2-VL-2B-Instruct")

There are no kerasformers release uploads for this family; the ``hf:`` path
(``config_from_hf`` + ``transfer_from_hf``) is canonical.
"""

import itertools

import keras
import numpy as np
from keras import layers, ops

from kerasformers.base import BaseModel

from .config import QWEN2_VL_CONFIG, QWEN2_VL_TOKENS, QWEN2_VL_WEIGHTS
from .qwen2_vl_layers import (
    Qwen2VLDecoderLayer,
    Qwen2VLPatchEmbed,
    Qwen2VLPatchMerger,
    Qwen2VLRMSNorm,
    Qwen2VLVisionBlock,
)

_MASK_NEG = -1e9


def vision_rotary_cos_sin(grid_thw, head_dim, spatial_merge_size, theta=10000.0):
    """2D vision rotary tables for the flattened patch sequence.

    Replicates HF ``Qwen2VisionTransformer.rot_pos_emb``: per image, height and
    width position ids are laid out in ``spatial_merge_size`` block order, the
    rotary table is gathered per position, and ``emb = cat(freqs, freqs)``.

    Args:
        grid_thw: iterable of ``(t, h, w)`` patch-grid sizes per image.
        head_dim: vision attention head dim (rotary covers all of it).
        spatial_merge_size: patch merge factor (block ordering).
        theta: rotary base.

    Returns:
        ``(cos, sin)`` numpy arrays, each ``(total_patches, head_dim)``.
    """
    m = spatial_merge_size
    rotary_dim = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
    )
    pos_ids = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        hpos = np.broadcast_to(np.arange(h)[:, None], (h, w))
        hpos = hpos.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).flatten()
        wpos = np.broadcast_to(np.arange(w)[None, :], (h, w))
        wpos = wpos.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).flatten()
        pos_ids.append(np.tile(np.stack([hpos, wpos], axis=-1), (t, 1)))
    pos_ids = np.concatenate(pos_ids, axis=0)

    max_grid = int(max(int(h) for _, h, w in grid_thw) | 0)
    max_grid = max(int(max(h, w)) for _, h, w in grid_thw)
    seq = np.arange(max_grid, dtype=np.float32)
    freqs_full = np.outer(seq, inv_freq)
    rotary = freqs_full[pos_ids].reshape(pos_ids.shape[0], -1)
    emb = np.concatenate([rotary, rotary], axis=-1)
    return np.cos(emb).astype("float32"), np.sin(emb).astype("float32")


def vision_block_mask(grid_thw):
    """Additive block-diagonal mask so patches only attend within their image.

    Returns ``None`` when there's a single block (full attention), else a
    ``(1, 1, total, total)`` float mask (0 within an image, ``_MASK_NEG`` across).
    """
    seqlens = [int(t) * int(h) * int(w) for t, h, w in grid_thw]
    if len(seqlens) <= 1:
        return None
    seg = np.concatenate([np.full(n, i) for i, n in enumerate(seqlens)])
    mask = np.where(seg[:, None] == seg[None, :], 0.0, _MASK_NEG).astype("float32")
    return mask[None, None]


def text_rope_cos_sin(position_ids, head_dim, theta):
    """Per-axis rotary tables from 3D position ids.

    Args:
        position_ids: ``(3, batch, seq)`` int array (temporal/height/width).
        head_dim: text attention head dim.
        theta: rotary base.

    Returns:
        ``(cos, sin)`` numpy arrays, each ``(3, batch, seq, head_dim)``.
    """
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = position_ids.astype("float32")[..., None] * inv_freq
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype("float32"), np.sin(emb).astype("float32")


def merge_mrope(table, mrope_section):
    """Collapse the 3 position axes into one per the M-RoPE channel sections.

    ``table`` is ``(3, batch, seq, head_dim)``; for the i-th channel chunk we
    keep the ``i % 3`` position axis. Returns ``(batch, seq, head_dim)``.
    """
    sections = list(mrope_section) * 2
    splits = np.split(table, np.cumsum(sections)[:-1], axis=-1)
    return np.concatenate([splits[i][i % 3] for i in range(len(splits))], axis=-1)


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2VLVisionModel(layers.Layer):
    """Vision tower: Conv3d-as-Dense patch embed -> rotary blocks -> 2x2 merger.

    ``call(pixel_values, grid_thw)`` takes the processor's flattened patches
    ``(num_patches, patch_dim)`` plus the per-image ``(t, h, w)`` grid and
    returns merged image embeddings ``(num_merged_tokens, llm_hidden)``.
    """

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        llm_hidden_size,
        mlp_ratio=4,
        spatial_merge_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.llm_hidden_size = llm_hidden_size
        self.mlp_ratio = mlp_ratio
        self.spatial_merge_size = spatial_merge_size
        self.head_dim = embed_dim // num_heads

        self.patch_embed = Qwen2VLPatchEmbed(embed_dim, name="patch_embed")
        self.blocks = [
            Qwen2VLVisionBlock(embed_dim, num_heads, mlp_ratio, name=f"blocks_{i}")
            for i in range(depth)
        ]
        self.merger = Qwen2VLPatchMerger(
            llm_hidden_size, embed_dim, spatial_merge_size, name="merger"
        )

    def call(self, pixel_values, grid_thw):
        cos, sin = vision_rotary_cos_sin(
            grid_thw, self.head_dim, self.spatial_merge_size
        )
        cos = ops.convert_to_tensor(cos)
        sin = ops.convert_to_tensor(sin)
        mask = vision_block_mask(grid_thw)
        if mask is not None:
            mask = ops.convert_to_tensor(mask)

        hidden = self.patch_embed(pixel_values)
        for block in self.blocks:
            hidden = block(hidden, cos, sin, attention_mask=mask)
        return self.merger(hidden)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "llm_hidden_size": self.llm_hidden_size,
                "mlp_ratio": self.mlp_ratio,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2VLTextModel(layers.Layer):
    """Qwen2 causal decoder: embed -> N decoder layers -> RMSNorm.

    Token embedding lives here (``embed_tokens``) and is reused (tied) as the
    LM head by :class:`Qwen2VLModel`. ``call`` accepts pre-computed
    ``inputs_embeds`` (the multimodal-fused sequence) and merged M-RoPE
    ``cos`` / ``sin``; it threads an optional KV cache for decoding.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=None,
        rms_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.rms_norm_eps = rms_norm_eps

        self.embed_tokens = layers.Embedding(
            vocab_size, hidden_size, name="embed_tokens"
        )
        self.decoder_layers = [
            Qwen2VLDecoderLayer(
                hidden_size,
                intermediate_size,
                num_attention_heads,
                num_key_value_heads,
                head_dim=self.head_dim,
                rms_norm_eps=rms_norm_eps,
                name=f"layers_{i}",
            )
            for i in range(num_hidden_layers)
        ]
        self.norm = Qwen2VLRMSNorm(eps=rms_norm_eps, name="norm")

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
        hidden = self.norm(hidden)
        return (hidden, new_cache) if use_cache else hidden

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
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2VLModel(BaseModel):
    """Qwen2-VL: vision tower + Qwen2 decoder fused by M-RoPE, with LM head.

    Call with a dict::

        out = model({
            "input_ids": (batch, seq),            # int, with image placeholders
            "pixel_values": (num_patches, pdim),  # flattened image patches
            "image_grid_thw": (num_images, 3),    # per-image (t, h, w)
            "attention_mask": (batch, seq),        # optional 1/0 padding mask
        })
        out["logits"]  # (batch, seq, vocab_size)

    Text-only inputs (no ``pixel_values`` / ``image_grid_thw``) are supported.
    """

    HF_MODEL_TYPE = "qwen2_vl"
    BASE_MODEL_CONFIG = QWEN2_VL_CONFIG
    BASE_WEIGHT_CONFIG = QWEN2_VL_WEIGHTS

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        mrope_section=(16, 24, 24),
        tie_word_embeddings=True,
        vision_depth=32,
        vision_embed_dim=1280,
        vision_num_heads=16,
        vision_mlp_ratio=4,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        image_token_id=QWEN2_VL_TOKENS["image_token_id"],
        video_token_id=QWEN2_VL_TOKENS["video_token_id"],
        vision_start_token_id=QWEN2_VL_TOKENS["vision_start_token_id"],
        vision_end_token_id=QWEN2_VL_TOKENS["vision_end_token_id"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.mrope_section = tuple(mrope_section)
        self.tie_word_embeddings = tie_word_embeddings
        self.vision_depth = vision_depth
        self.vision_embed_dim = vision_embed_dim
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_ratio = vision_mlp_ratio
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

        self.visual = Qwen2VLVisionModel(
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            llm_hidden_size=hidden_size,
            mlp_ratio=vision_mlp_ratio,
            spatial_merge_size=spatial_merge_size,
            name="visual",
        )
        self.language_model = Qwen2VLTextModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=rms_norm_eps,
            name="language_model",
        )

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
        """3D position ids ``(3, batch, seq)`` and ``rope_deltas`` ``(batch,)``."""
        m = self.spatial_merge_size
        input_ids = np.asarray(input_ids)
        batch, seq = input_ids.shape
        position_ids = np.zeros((3, batch, seq), dtype=np.int64)
        rope_deltas = []
        img_iter = iter(image_grid_thw) if image_grid_thw is not None else None

        for bi in range(batch):
            ids = input_ids[bi]
            keep = (
                np.asarray(attention_mask[bi]).astype(bool)
                if attention_mask is not None
                else np.ones(seq, dtype=bool)
            )
            ids_kept = ids[keep]
            ttype = np.where(ids_kept == self.image_token_id, 1, 0)
            ttype = np.where(ids_kept == self.video_token_id, 2, ttype)

            cur = 0
            pieces = []
            for key, group in itertools.groupby(
                enumerate(ttype.tolist()), lambda x: x[1]
            ):
                g = list(group)
                start, end = g[0][0], g[-1][0] + 1
                if key == 0:
                    length = end - start
                    pieces.append(
                        np.broadcast_to(np.arange(length) + cur, (3, length)).copy()
                    )
                    cur += length
                else:
                    t, h, w = (int(v) for v in next(img_iter))
                    lt, lh, lw = t, h // m, w // m
                    n = lt * lh * lw
                    wpos = np.tile(np.arange(cur, cur + lw), lh * lt)
                    hpos = np.repeat(np.arange(cur, cur + lh), lw * lt)
                    tpos = np.full(n, cur * self.tokens_per_second)
                    pieces.append(np.stack([tpos, hpos, wpos], axis=0))
                    cur += max(h, w) // m
            llm_positions = np.concatenate(pieces, axis=1)
            position_ids[:, bi, keep] = llm_positions
            rope_deltas.append(int(llm_positions.max()) + 1 - int(ids_kept.shape[0]))
        return position_ids, np.asarray(rope_deltas, dtype=np.int64)

    def embed_tokens(self, input_ids):
        return self.language_model.embed_tokens(input_ids)

    def get_image_features(self, pixel_values, image_grid_thw):
        """Run the vision tower -> merged image embeddings ``(n_tokens, hidden)``."""
        return self.visual(pixel_values, image_grid_thw)

    def _causal_mask(self, q_len, kv_len, offset):
        """Additive causal mask ``(1, 1, q_len, kv_len)`` (query i sees key<=i+offset)."""
        qi = np.arange(q_len)[:, None] + offset
        ki = np.arange(kv_len)[None, :]
        mask = np.where(ki <= qi, 0.0, _MASK_NEG).astype("float32")
        return ops.convert_to_tensor(mask[None, None])

    def _forward_features(self, inputs):
        """Run vision + fusion + decoder -> fused hidden state ``(B, L, hidden)``."""
        if not isinstance(inputs, dict):
            raise ValueError(f"{type(self).__name__} expects a dict of inputs.")
        input_ids_np = np.asarray(ops.convert_to_numpy(inputs["input_ids"])).astype(
            "int64"
        )
        seq = input_ids_np.shape[1]
        inputs_embeds, position_ids, _, extra = self._prepare_inputs(
            input_ids_np,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("attention_mask"),
        )
        cos, sin = self._merged_cos_sin(position_ids)
        attn_mask = self._causal_mask(seq, seq, offset=0)
        return self.language_model(
            inputs_embeds, cos, sin, attention_mask=attn_mask, **extra
        )

    def call(self, inputs):
        """Return raw fused features. Use ``Qwen2VLGenerate`` for logits/text."""
        return {"last_hidden_state": self._forward_features(inputs)}

    def _prepare_inputs(
        self, input_ids_np, pixel_values, image_grid_thw, attention_mask
    ):
        """Build the multimodal-fused ``inputs_embeds`` + 3D ``position_ids``.

        Returns ``(inputs_embeds, position_ids, rope_deltas, extra)``, where
        ``extra`` is a dict of additional ``language_model`` call kwargs (empty
        here; Qwen3-VL uses it to thread DeepStack features). Image features
        (if any) are scattered into the ``image_token_id`` placeholder slots,
        and M-RoPE positions come from :meth:`get_rope_index`.
        """
        batch, seq = input_ids_np.shape
        inputs_embeds = self.language_model.embed_tokens(
            ops.convert_to_tensor(input_ids_np)
        )
        rope_deltas = np.zeros((batch,), dtype=np.int64)

        if pixel_values is not None and image_grid_thw is not None:
            grid = np.asarray(ops.convert_to_numpy(image_grid_thw)).astype("int64")
            image_embeds = self.get_image_features(pixel_values, grid)
            flat_mask = (input_ids_np == self.image_token_id).reshape(-1)
            embeds_flat = ops.reshape(inputs_embeds, (batch * seq, self.hidden_size))
            idx = np.nonzero(flat_mask)[0]
            embeds_flat = ops.scatter_update(
                embeds_flat,
                np.expand_dims(idx, -1).astype("int32"),
                ops.cast(image_embeds, embeds_flat.dtype),
            )
            inputs_embeds = ops.reshape(embeds_flat, (batch, seq, self.hidden_size))
            position_ids, rope_deltas = self.get_rope_index(
                input_ids_np, grid, attention_mask
            )
        else:
            am = (
                None
                if attention_mask is None
                else np.asarray(ops.convert_to_numpy(attention_mask))
            )
            if am is not None:
                pos = np.cumsum(am, axis=-1) - 1
                pos = np.where(am == 0, 0, pos)
            else:
                pos = np.broadcast_to(np.arange(seq), (batch, seq))
            position_ids = np.broadcast_to(pos, (3, batch, seq)).copy()
        return inputs_embeds, position_ids, rope_deltas, {}

    def _merged_cos_sin(self, position_ids):
        cos3, sin3 = text_rope_cos_sin(position_ids, self.head_dim, self.rope_theta)
        cos = ops.convert_to_tensor(merge_mrope(cos3, self.mrope_section))
        sin = ops.convert_to_tensor(merge_mrope(sin3, self.mrope_section))
        return cos, sin

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config.get("vision_config", {})
        rope_scaling = hf_config.get("rope_scaling") or {}
        mrope = rope_scaling.get("mrope_section", [16, 24, 24])
        return {
            "vocab_size": hf_config["vocab_size"],
            "hidden_size": hf_config["hidden_size"],
            "intermediate_size": hf_config["intermediate_size"],
            "num_hidden_layers": hf_config["num_hidden_layers"],
            "num_attention_heads": hf_config["num_attention_heads"],
            "num_key_value_heads": hf_config["num_key_value_heads"],
            "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            "mrope_section": tuple(mrope),
            "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
            "vision_depth": vc.get("depth", 32),
            "vision_embed_dim": vc.get("embed_dim", 1280),
            "vision_num_heads": vc.get("num_heads", 16),
            "vision_mlp_ratio": vc.get("mlp_ratio", 4),
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
        from .convert_qwen2_vl_hf_to_keras import transfer_qwen2_vl_weights

        transfer_qwen2_vl_weights(keras_model, hf_state_dict)

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
                "rms_norm_eps": self.rms_norm_eps,
                "rope_theta": self.rope_theta,
                "mrope_section": self.mrope_section,
                "tie_word_embeddings": self.tie_word_embeddings,
                "vision_depth": self.vision_depth,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_num_heads": self.vision_num_heads,
                "vision_mlp_ratio": self.vision_mlp_ratio,
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


class _QwenVLGenerateMixin:
    """Adds an LM head + greedy multimodal ``.generate()`` on top of a base
    Qwen-VL model.

    The base model supplies the feature machinery (``_prepare_inputs`` ->
    ``(inputs_embeds, position_ids, rope_deltas, extra)``, ``_merged_cos_sin``,
    ``_causal_mask``, ``language_model``). ``extra`` lets Qwen3-VL thread its
    DeepStack tensors through the prefill; decode steps are text-only so they
    pass no ``extra``. Generation is therefore shared across all three families.
    """

    def _init_lm_head(self):
        self.lm_head = (
            None
            if self.tie_word_embeddings
            else layers.Dense(self.vocab_size, use_bias=False, name="lm_head")
        )

    def _lm_logits(self, hidden):
        if getattr(self, "lm_head", None) is not None:
            return self.lm_head(hidden)
        emb = self.language_model.embed_tokens.embeddings
        return ops.matmul(hidden, ops.transpose(emb))

    def call(self, inputs):
        hidden = self._forward_features(inputs)
        return {"logits": self._lm_logits(hidden), "last_hidden_state": hidden}

    def generate(
        self,
        input_ids,
        pixel_values=None,
        image_grid_thw=None,
        attention_mask=None,
        max_new_tokens=128,
        eos_token_id=(151645,),
        return_ids=True,
    ):
        """Greedy decoding with a KV cache and incremental M-RoPE.

        Each new token's position is ``cache_len + rope_delta`` on all three
        axes; the per-layer ``(k, v)`` cache is threaded forward so only the new
        token is recomputed. Returns ``(batch, num_new_tokens)`` ids (numpy).
        """
        input_ids_np = np.asarray(ops.convert_to_numpy(input_ids)).astype("int64")
        batch, prompt_len = input_ids_np.shape
        inputs_embeds, position_ids, rope_deltas, extra = self._prepare_inputs(
            input_ids_np, pixel_values, image_grid_thw, attention_mask
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
            pos = np.broadcast_to(
                (cur_len + rope_deltas).reshape(1, batch, 1), (3, batch, 1)
            ).copy()
            step_cos, step_sin = self._merged_cos_sin(pos)
            step_embeds = self.language_model.embed_tokens(
                ops.convert_to_tensor(next_tok)
            )
            hidden, cache = self.language_model(
                step_embeds,
                step_cos,
                step_sin,
                attention_mask=None,
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


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2VLGenerate(_QwenVLGenerateMixin, Qwen2VLModel):
    """Qwen2-VL with an LM head + greedy ``.generate()`` (image+text -> text)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_lm_head()
