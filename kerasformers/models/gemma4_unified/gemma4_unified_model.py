import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel
from kerasformers.models.gemma4.gemma4_model import Gemma4Model
from kerasformers.models.gemma4.gemma4_vision_layers import Gemma4MultimodalEmbedder

from .gemma4_unified_config import (
    GEMMA4_UNIFIED_GENERATE_CONFIG,
    GEMMA4_UNIFIED_GENERATE_WEIGHTS_URLS,
)
from .gemma4_unified_vision import Gemma4UnifiedVisionEmbedder

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4UnifiedModel(SubclassedBaseModel):
    """Gemma 4 unified vision + audio + text backbone (no LM head).

    The unified checkpoints (google/gemma-4-12B) are encoder-free: instead of the
    "gemma4" family's NaViT vision tower and USM audio conformer, images arrive as
    raw 48px merged pixel patches projected by :class:`Gemma4UnifiedVisionEmbedder`
    (Dense + factorized 2D position embedding), and audio arrives as raw 640-sample
    waveform frames projected straight through the shared
    :class:`Gemma4MultimodalEmbedder` (weightless RMSNorm then Dense). Both feed
    their soft tokens onto the ``image_token_id`` / ``audio_token_id`` slots of the
    prompt. The text tower is the plain dense Gemma 4 decoder
    (:class:`Gemma4Model` with no Per-Layer Embeddings and no MoE) with global
    ``K = V`` attention and per-layer scalars. On the sliding-window layers the
    image soft tokens attend bidirectionally within their block (Gemma 4's
    ``vision`` bidirectional setting); global layers stay causal. Returns raw text
    features; the LM head lives in :class:`Gemma4UnifiedGenerate`.

    Args:
        text_config: Keyword arguments forwarded to :class:`Gemma4Model`.
        vision_config: Encoder-free vision embedder settings (or ``None``).
        audio_config: Encoder-free audio embedder settings (or ``None``).
        image_token_id: Prompt token id whose slots receive image soft tokens.
        video_token_id: Prompt token id whose slots receive video soft tokens.
        audio_token_id: Prompt token id whose slots receive audio soft tokens.
        pad_token_id: Token id used to embed multimodal slots before scatter.
        use_bidirectional_vision: Enable blockwise bidirectional vision masking.
    """

    HF_MODEL_TYPE = ("gemma4_unified",)
    BASE_MODEL_CONFIG = GEMMA4_UNIFIED_GENERATE_CONFIG
    BASE_WEIGHT_CONFIG = GEMMA4_UNIFIED_GENERATE_WEIGHTS_URLS
    default_load_dtype = "bfloat16"  # Google ships gemma-4 in bf16

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        image_token_id=258880,
        video_token_id=258884,
        audio_token_id=258881,
        pad_token_id=0,
        use_bidirectional_vision=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_config = dict(text_config or {})
        self.vision_config = dict(vision_config) if vision_config else None
        self.audio_config = dict(audio_config) if audio_config else None
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.pad_token_id = pad_token_id
        self.use_bidirectional_vision = use_bidirectional_vision

        self.language_model = Gemma4Model(**self.text_config, name="language_model")
        text_hidden = self.language_model.embed_dim
        eps = self.language_model.norm_eps

        # Encoder-free towers: both optional (with neither, this is a plain text
        # generator, matching Gemma4UnifiedForConditionalGeneration on text input).
        self.embed_vision = None
        self.patch_dim = 0
        if self.vision_config is not None:
            model_patch_size = (
                self.vision_config["patch_size"]
                * self.vision_config["pooling_kernel_size"]
            )
            self.patch_dim = model_patch_size**2 * 3
            self.embed_vision = Gemma4UnifiedVisionEmbedder(
                patch_dim=self.patch_dim,
                mm_embed_dim=self.vision_config["mm_embed_dim"],
                mm_posemb_size=self.vision_config["mm_posemb_size"],
                text_hidden_size=text_hidden,
                eps=self.vision_config.get("eps", eps),
                name="embed_vision",
            )
        self.embed_audio = None
        self.audio_dim = 0
        if self.audio_config is not None:
            self.audio_dim = self.audio_config["audio_embed_dim"]
            self.embed_audio = Gemma4MultimodalEmbedder(
                text_hidden,
                eps=self.audio_config.get("eps", eps),
                name="embed_audio",
            )

    def build_for_transfer(self):
        # Materialize every sublayer weight before a weight transfer: one text
        # token, one image slot (with a single valid patch) and one audio slot
        # (with a single valid frame) so both encoder-free towers build.
        input_ids = [ops.zeros((1, 1), dtype="int32")]
        inputs = {}
        if self.embed_vision is not None:
            input_ids.append(ops.full((1, 1), self.image_token_id, dtype="int32"))
            inputs["pixel_values"] = ops.zeros((1, 1, self.patch_dim), dtype="float32")
            inputs["pixel_position_ids"] = ops.zeros((1, 1, 2), dtype="int32")
        if self.embed_audio is not None:
            input_ids.append(ops.full((1, 1), self.audio_token_id, dtype="int32"))
            inputs["input_features"] = ops.zeros(
                (1, 1, self.audio_dim), dtype="float32"
            )
            inputs["input_features_mask"] = ops.ones((1, 1), dtype="bool")
        inputs["input_ids"] = ops.concatenate(input_ids, axis=1)
        self(inputs)

    def scatter_soft_tokens(self, text_embeds, slot_mask, features):
        # Replace every True position of slot_mask (row-major over batch then
        # sequence) with successive rows of features, mirroring HF's
        # masked_scatter. features is [num_soft_tokens, hidden].
        shape = ops.shape(text_embeds)
        flat_mask = ops.reshape(slot_mask, (-1,))
        rank = ops.cumsum(ops.cast(flat_mask, "int32")) - 1
        rank = ops.clip(rank, 0, ops.shape(features)[0] - 1)
        gathered = ops.take(features, rank, axis=0)
        gathered = ops.reshape(gathered, shape)
        return ops.where(ops.expand_dims(slot_mask, -1), gathered, text_embeds)

    def compact_valid(self, features, valid_mask):
        # Gather the valid (non-padding) soft tokens to the front in row-major
        # (batch, token) order, mirroring HF's boolean-index padding strip.
        shape = ops.shape(features)
        flat = ops.reshape(features, (-1, shape[-1]))
        vmask = ops.reshape(valid_mask, (-1,))
        n = ops.shape(flat)[0]
        rank = ops.cumsum(ops.cast(vmask, "int32")) - 1
        target = ops.where(vmask, rank, n)
        buffer = ops.zeros((n + 1, shape[-1]), dtype=flat.dtype)
        buffer = ops.scatter_update(buffer, target[:, None], flat)
        return buffer[:n]

    def block_sequence_ids(self, is_vision):
        # Assign a per-image block id to contiguous vision runs; -1 for text.
        zeros = ops.zeros_like(is_vision[:, :1])
        prev = ops.concatenate([zeros, is_vision[:, :-1]], axis=1)
        new_starts = ops.logical_and(is_vision, ops.logical_not(prev))
        group = ops.cumsum(ops.cast(new_starts, "int32"), axis=1) - 1
        return ops.where(is_vision, group, ops.full_like(group, -1))

    def fuse_embeds(
        self,
        input_ids,
        pixel_values=None,
        pixel_position_ids=None,
        input_features=None,
        input_features_mask=None,
    ):
        # Embed text (multimodal slots replaced by pad), then scatter the
        # projected, padding-stripped vision and audio soft tokens onto their
        # placeholder slots. Returns fused embeddings and the vision-token mask.
        lm = self.language_model
        is_image = input_ids == self.image_token_id
        is_video = input_ids == self.video_token_id
        is_audio = input_ids == self.audio_token_id
        is_vision = ops.logical_or(is_image, is_video)
        multimodal = ops.logical_or(is_vision, is_audio)

        hidden = lm.embed_scaled(ops.where(multimodal, self.pad_token_id, input_ids))

        if pixel_values is not None and self.embed_vision is not None:
            positions = ops.cast(ops.convert_to_tensor(pixel_position_ids), "int32")
            feats = self.embed_vision(ops.convert_to_tensor(pixel_values), positions)
            valid = ops.logical_not(ops.all(positions == -1, axis=-1))
            feats = self.compact_valid(feats, valid)
            hidden = self.scatter_soft_tokens(
                hidden, is_image, ops.cast(feats, hidden.dtype)
            )

        if input_features is not None and self.embed_audio is not None:
            audio_feats = self.embed_audio(ops.convert_to_tensor(input_features))
            if input_features_mask is not None:
                valid = ops.cast(ops.convert_to_tensor(input_features_mask), "bool")
                feats = self.compact_valid(audio_feats, valid)
            else:
                feats = ops.reshape(audio_feats, (-1, lm.embed_dim))
            hidden = self.scatter_soft_tokens(
                hidden, is_audio, ops.cast(feats, hidden.dtype)
            )
        return hidden, is_vision

    def prefill_rope_masks(self, is_vision, attention_mask, batch, seq):
        lm = self.language_model
        position_ids = lm.compute_position_ids(attention_mask, batch, seq)
        cos_l, sin_l = lm.rope_tables(position_ids, local=True)
        cos_g, sin_g = lm.rope_tables(position_ids, local=False)
        block_ids = (
            self.block_sequence_ids(is_vision)
            if self.use_bidirectional_vision
            else None
        )
        full_mask, sliding_mask = lm.build_masks(seq, attention_mask, block_ids)
        return (cos_l, sin_l, cos_g, sin_g), (full_mask, sliding_mask)

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        lm = self.language_model
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        attention_mask = inputs.get("attention_mask")
        hidden, is_vision = self.fuse_embeds(
            input_ids,
            inputs.get("pixel_values"),
            inputs.get("pixel_position_ids"),
            inputs.get("input_features"),
            inputs.get("input_features_mask"),
        )
        rope, masks = self.prefill_rope_masks(is_vision, attention_mask, batch, seq)
        hidden = lm.run_layers(hidden, *rope, *masks)
        return {"last_hidden_state": hidden}

    @staticmethod
    def vision_config_from_hf(vision):
        return {
            "patch_size": vision.get("patch_size", 16),
            "pooling_kernel_size": vision.get("pooling_kernel_size", 3),
            "mm_embed_dim": vision.get("mm_embed_dim", 3840),
            "mm_posemb_size": vision.get("mm_posemb_size", 1120),
            "output_proj_dims": vision.get("output_proj_dims", 3840),
            "eps": vision.get("rms_norm_eps", 1e-6),
        }

    @staticmethod
    def audio_config_from_hf(audio):
        return {
            "audio_embed_dim": audio.get("audio_embed_dim", 640),
            "output_proj_dims": audio.get("output_proj_dims", 640),
            "eps": audio.get("rms_norm_eps", 1e-6),
        }

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config["text_config"]
        vision = hf_config.get("vision_config")
        audio = hf_config.get("audio_config")
        vision_ok = bool(vision) and vision.get("model_type") == "gemma4_unified_vision"
        audio_ok = bool(audio) and audio.get("model_type") == "gemma4_unified_audio"
        return {
            "text_config": Gemma4Model.config_from_hf(hf_config),
            "vision_config": cls.vision_config_from_hf(vision) if vision_ok else None,
            "audio_config": cls.audio_config_from_hf(audio) if audio_ok else None,
            "image_token_id": hf_config.get("image_token_id", 258880),
            "video_token_id": hf_config.get("video_token_id", 258884),
            "audio_token_id": hf_config.get("audio_token_id", 258881),
            "pad_token_id": text.get("pad_token_id", 0),
            "use_bidirectional_vision": text.get("use_bidirectional_attention")
            == "vision",
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma4_unified_hf_to_keras import transfer_gemma4_unified_weights

        transfer_gemma4_unified_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "text_config": self.text_config,
                "vision_config": self.vision_config,
                "audio_config": self.audio_config,
                "image_token_id": self.image_token_id,
                "video_token_id": self.video_token_id,
                "audio_token_id": self.audio_token_id,
                "pad_token_id": self.pad_token_id,
                "use_bidirectional_vision": self.use_bidirectional_vision,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4UnifiedGenerate(Gemma4UnifiedModel, BaseGeneration):
    """Gemma 4 unified backbone + a (tied) LM head with fast ``.generate()``.

    The single generation entry point, mirroring transformers'
    ``Gemma4UnifiedForConditionalGeneration``: it drives the encoder-free
    multimodal 12B and any text-only unified checkpoint through the same API.
    When a vision or audio tower is present the prefill fuses the soft tokens and
    applies the blockwise vision mask; text-only prompts skip straight to the text
    decoder. Decoding is always text-only and reuses the per-layer sliding /
    global K/V cache geometry. Pass ``pixel_values`` / ``pixel_position_ids`` /
    ``input_features`` / ``input_features_mask`` as keyword prefill inputs to
    ``generate`` when the checkpoint has the towers.
    """

    HF_MODEL_TYPE = ("gemma4_unified", "gemma4_unified_text")
    BASE_MODEL_CONFIG = GEMMA4_UNIFIED_GENERATE_CONFIG
    BASE_WEIGHT_CONFIG = GEMMA4_UNIFIED_GENERATE_WEIGHTS_URLS
    default_load_dtype = "bfloat16"  # Google ships gemma-4 in bf16

    eos_token_id = (1, 106)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lm = self.language_model
        self.lm_head = (
            None
            if lm.tie_embeddings
            else layers.Dense(lm.vocab_size, use_bias=False, name="lm_head")
        )

    def project(self, hidden):
        lm = self.language_model
        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = ops.matmul(hidden, ops.transpose(lm.token_embedding.embeddings))
        if lm.final_logit_softcapping is not None:
            cap = lm.final_logit_softcapping
            logits = ops.tanh(logits / cap) * cap
        return logits

    def call(self, inputs):
        hidden = super().call(inputs)["last_hidden_state"]
        return {"logits": self.project(hidden), "last_hidden_state": hidden}

    def build_cache(
        self,
        token_ids,
        padding_mask,
        max_len,
        pixel_values=None,
        pixel_position_ids=None,
        input_features=None,
        input_features_mask=None,
    ):
        lm = self.language_model
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        token_ids = ops.cast(ops.convert_to_tensor(token_ids), "int32")
        hidden, is_vision = self.fuse_embeds(
            token_ids,
            pixel_values,
            pixel_position_ids,
            input_features,
            input_features_mask,
        )
        rope, masks = self.prefill_rope_masks(
            is_vision, padding_mask, batch, prompt_len
        )
        cos_l, sin_l, cos_g, sin_g = rope
        full_mask, sliding_mask = masks
        layer_caches = []
        shared_kv = {}  # layer_type -> storing layer's prompt-length (k, v)
        shared_stacked = {}  # layer_type -> storing layer's padded [ck, cv]
        for i, layer in enumerate(lm.decoder_layers):
            sliding = lm.is_sliding(i)
            layer_type = "sliding" if sliding else "global"
            cos, sin, mask = (
                (cos_l, sin_l, sliding_mask) if sliding else (cos_g, sin_g, full_mask)
            )
            is_shared = lm.num_kv_shared_layers > 0 and i >= lm.first_kv_shared
            hidden, (k, v) = layer(
                hidden,
                cos,
                sin,
                attention_mask=mask,
                shared_kv=shared_kv.get(layer_type) if is_shared else None,
            )
            if is_shared:
                layer_caches.append(shared_stacked[layer_type])
                continue
            nkv = int(k.shape[1])
            hd = int(k.shape[3])
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            stacked = ops.stack([ck, cv], axis=1)
            layer_caches.append(stacked)
            if lm.num_kv_shared_layers > 0:
                shared_kv[layer_type] = (k, v)
                shared_stacked[layer_type] = stacked
        logits = self.project(lm.final_norm(hidden)[:, -1, :])
        return tuple(layer_caches), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        lm = self.language_model
        batch = int(token_ids.shape[0])
        max_len = int(cache[0].shape[3])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos_l, sin_l = lm.rope_tables(positions, local=True)
        cos_g, sin_g = lm.rope_tables(positions, local=False)
        ar = ops.arange(max_len)
        full_km = ops.cast(ops.where(ar <= pos, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        sliding_km = ops.cast(
            ops.where(
                ops.logical_and(ar <= pos, ar > pos - lm.sliding_window),
                0.0,
                MASK_NEG,
            ),
            "float32",
        )[None, None, None, :]
        token_ids = ops.cast(token_ids, "int32")
        h = lm.embed_scaled(token_ids)
        new_caches = []
        shared_stacked = {}  # layer_type -> storing layer's updated [ck, cv]
        for i, layer in enumerate(lm.decoder_layers):
            sliding = lm.is_sliding(i)
            layer_type = "sliding" if sliding else "global"
            cos, sin, km = (
                (cos_l, sin_l, sliding_km) if sliding else (cos_g, sin_g, full_km)
            )
            is_shared = lm.num_kv_shared_layers > 0 and i >= lm.first_kv_shared
            if is_shared:
                stacked = shared_stacked[layer_type]
                h, _, _ = layer.decode_step(
                    h, cos, sin, stacked[:, 0], stacked[:, 1], pos, km
                )
                new_caches.append(stacked)
                continue
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[i][:, 0], cache[i][:, 1], pos, km
            )
            stacked = ops.stack([ck, cv], axis=1)
            new_caches.append(stacked)
            if lm.num_kv_shared_layers > 0:
                shared_stacked[layer_type] = stacked
        logits = self.project(lm.final_norm(h))[:, 0, :]
        return logits, tuple(new_caches)
