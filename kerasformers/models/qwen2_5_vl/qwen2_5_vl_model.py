"""Qwen2.5-VL — Qwen2-VL with a windowed-attention RMSNorm/SwiGLU vision tower.

The text decoder and the multimodal fusion / M-RoPE / generation are inherited
unchanged from :class:`Qwen2VLModel`; only the vision tower is replaced. For
image inputs the position logic is identical to Qwen2-VL (the extra
``tokens_per_second`` only rescales *video* temporal positions).

    model = Qwen2_5_VLModel.from_weights("hf:Qwen/Qwen2.5-VL-3B-Instruct")
"""

import keras
import numpy as np
from keras import layers, ops

from kerasformers.models.qwen2_vl.qwen2_vl_model import (
    _MASK_NEG,
    Qwen2VLModel,
    _QwenVLGenerateMixin,
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

    Returns ``(window_index, cu_window_seqlens)``: a permutation of the
    ``seq // merge_unit`` merge-unit groups into window-contiguous order, and
    cumulative per-window sequence lengths (in patch units).
    """
    m = spatial_merge_size
    merge_unit = m * m
    vit_window = window_size // m // patch_size
    window_index = []
    cu_window_seqlens = [0]
    offset = 0
    for t, h, w in np.asarray(grid_thw).tolist():
        lh, lw = h // m, w // m
        index = np.arange(t * lh * lw).reshape(t, lh, lw)
        pad_h = (vit_window - lh % vit_window) % vit_window
        pad_w = (vit_window - lw % vit_window) % vit_window
        nwh = (lh + pad_h) // vit_window
        nww = (lw + pad_w) // vit_window
        index_padded = np.pad(
            index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100
        )
        index_padded = index_padded.reshape(t, nwh, vit_window, nww, vit_window)
        index_padded = index_padded.transpose(0, 1, 3, 2, 4).reshape(
            t, nwh * nww, vit_window, vit_window
        )
        seqlens = (index_padded != -100).sum((2, 3)).reshape(-1)
        index_flat = index_padded.reshape(-1)
        index_new = index_flat[index_flat != -100]
        window_index.append(index_new + offset)
        cu = np.cumsum(seqlens) * merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu.tolist())
        offset += t * lh * lw
    window_index = np.concatenate(window_index, axis=0)
    cu = np.array(cu_window_seqlens, dtype=np.int64)
    cu = cu[np.concatenate([[True], np.diff(cu) != 0])]
    return window_index, cu


def _segment_mask(cu_seqlens, total):
    """Additive block-diagonal mask (1,1,total,total) from cumulative seqlens."""
    seg = np.zeros(total, dtype=np.int64)
    for i in range(len(cu_seqlens) - 1):
        seg[int(cu_seqlens[i]) : int(cu_seqlens[i + 1])] = i
    mask = np.where(seg[:, None] == seg[None, :], 0.0, _MASK_NEG).astype("float32")
    return mask[None, None]


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
        grid = np.asarray(grid_thw).astype("int64")
        u = self.merge_unit
        seq = int(np.prod(grid, axis=1).sum())

        hidden = self.patch_embed(pixel_values)

        cos, sin = vision_rotary_cos_sin(grid, self.head_dim, self.spatial_merge_size)
        window_index, cu_window = get_window_index(
            grid, self.window_size, self.spatial_merge_size, self.patch_size
        )

        hidden = ops.reshape(hidden, (seq // u, u, self.embed_dim))
        hidden = ops.take(hidden, window_index, axis=0)
        hidden = ops.reshape(hidden, (seq, self.embed_dim))
        cos = cos.reshape(seq // u, u, self.head_dim)[window_index].reshape(seq, -1)
        sin = sin.reshape(seq // u, u, self.head_dim)[window_index].reshape(seq, -1)
        cos_t = ops.convert_to_tensor(cos)
        sin_t = ops.convert_to_tensor(sin)

        cu_full = np.concatenate(
            [[0], np.cumsum(np.repeat(grid[:, 1] * grid[:, 2], grid[:, 0]))]
        )
        full_mask = _segment_mask(cu_full, seq)
        window_mask = _segment_mask(cu_window, seq)
        full_mask = None if len(cu_full) <= 2 else ops.convert_to_tensor(full_mask)
        window_mask_t = ops.convert_to_tensor(window_mask)

        for i, block in enumerate(self.blocks):
            mask = full_mask if i in self.fullatt_block_indexes else window_mask_t
            hidden = block(hidden, cos_t, sin_t, attention_mask=mask)

        merged = self.merger(hidden)
        reverse = np.argsort(window_index)
        merged = ops.take(merged, reverse, axis=0)
        return merged

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
            Qwen2_5_VLDecoderLayer(
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
        self.norm = Qwen2_5_VLRMSNorm(eps=rms_norm_eps, name="norm")

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
class Qwen2_5_VLModel(Qwen2VLModel):
    """Qwen2.5-VL: Qwen2-VL text/fusion/generation with a windowed vision tower."""

    HF_MODEL_TYPE = "qwen2_5_vl"
    BASE_MODEL_CONFIG = QWEN2_5_VL_CONFIG
    BASE_WEIGHT_CONFIG = QWEN2_5_VL_WEIGHTS

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        mrope_section=(16, 24, 24),
        tie_word_embeddings=True,
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
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_heads = vision_num_heads
        self.vision_out_hidden_size = vision_out_hidden_size or hidden_size
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
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=rms_norm_eps,
            name="language_model",
        )

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
class Qwen2_5_VLGenerate(_QwenVLGenerateMixin, Qwen2_5_VLModel):
    """Qwen2.5-VL with an LM head + greedy ``.generate()`` (image+text -> text)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_lm_head()
