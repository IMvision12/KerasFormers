import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .gemma4_audio_layers import Gemma4AudioModel
from .gemma4_config import GEMMA4_MULTIMODAL_CONFIG, GEMMA4_MULTIMODAL_WEIGHTS_URLS
from .gemma4_model import MASK_NEG, Gemma4Model
from .gemma4_vision_layers import Gemma4MultimodalEmbedder, Gemma4VisionModel


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4MultimodalModel(SubclassedBaseModel):
    """Gemma 4 vision + text backbone (no LM head).

    Composes the NaViT vision tower (:class:`Gemma4VisionModel`), the soft-token
    projector (:class:`Gemma4MultimodalEmbedder`) and the text decoder
    (:class:`Gemma4Model`). Image patches become pooled soft tokens, are
    projected into the text embedding space and scattered onto the
    ``image_token_id`` slots of the prompt. On the sliding-window layers those
    soft tokens attend bidirectionally within their image block (the ``vision``
    setting of Gemma 4's ``use_bidirectional_attention``); the global layers stay
    strictly causal. Returns raw text features; the LM head lives in
    :class:`~kerasformers.models.gemma4.gemma4_model.Gemma4Generate`.

    Args:
        text_config: Keyword arguments forwarded to :class:`Gemma4Model`.
        vision_config: Keyword arguments forwarded to :class:`Gemma4VisionModel`.
        image_token_id: Prompt token id whose slots receive image soft tokens.
        video_token_id: Prompt token id whose slots receive video soft tokens.
        audio_token_id: Prompt token id marking audio soft-token slots.
        pad_token_id: Token id used to embed multimodal slots before scatter.
        use_bidirectional_vision: Enable blockwise bidirectional vision masking.
    """

    HF_MODEL_TYPE = ("gemma4",)
    BASE_MODEL_CONFIG = GEMMA4_MULTIMODAL_CONFIG
    BASE_WEIGHT_CONFIG = GEMMA4_MULTIMODAL_WEIGHTS_URLS

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
        self.vision_config = dict(vision_config or {})
        self.audio_config = dict(audio_config) if audio_config else None
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.pad_token_id = pad_token_id
        self.use_bidirectional_vision = use_bidirectional_vision

        self.language_model = Gemma4Model(**self.text_config, name="language_model")
        self.vision_model = Gemma4VisionModel(**self.vision_config, name="vision_tower")
        self.embed_vision = Gemma4MultimodalEmbedder(
            self.language_model.embed_dim,
            eps=self.language_model.norm_eps,
            name="embed_vision",
        )
        self.audio_tower = None
        self.embed_audio = None
        if self.audio_config is not None:
            self.audio_tower = Gemma4AudioModel(**self.audio_config, name="audio_tower")
            self.embed_audio = Gemma4MultimodalEmbedder(
                self.language_model.embed_dim,
                eps=self.language_model.norm_eps,
                name="embed_audio",
            )

    def build_for_transfer(self):
        # Instantiate every sublayer weight (text + vision + embed_vision) with a
        # minimal image so a weight transfer can run before the first real call.
        patch = self.vision_config.get("patch_size", 16)
        pool = self.vision_config.get("pooling_kernel_size", 3)
        num_patches = pool * pool
        pixel_values = ops.zeros((1, num_patches, 3 * patch * patch), dtype="float32")
        coords = ops.stack(
            ops.meshgrid(ops.arange(pool), ops.arange(pool), indexing="xy"), axis=-1
        )
        coords = ops.reshape(coords, (1, num_patches, 2))
        inputs = {
            "input_ids": ops.concatenate(
                [
                    ops.zeros((1, 1), dtype="int32"),
                    ops.full((1, 1), self.image_token_id, dtype="int32"),
                ],
                axis=1,
            ),
            "pixel_values": pixel_values,
            "pixel_position_ids": coords,
        }
        self(inputs)
        if self.audio_tower is not None:
            channels = self.audio_config.get("conv_channels", (128, 32))
            chunk = self.audio_config.get("chunk_size", 12)
            frames = 4 * chunk
            audio_ids = ops.concatenate(
                [
                    ops.zeros((1, 1), dtype="int32"),
                    ops.full((1, chunk), self.audio_token_id, dtype="int32"),
                ],
                axis=1,
            )
            self(
                {
                    "input_ids": audio_ids,
                    "input_features": ops.zeros(
                        (1, frames, channels[0]), dtype="float32"
                    ),
                    "input_features_mask": ops.ones((1, frames), dtype="bool"),
                }
            )

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
        # Gather the valid (non-padding) audio frames to the front in row-major
        # (batch, frame) order, mirroring HF's boolean-index padding strip.
        shape = ops.shape(features)
        flat = ops.reshape(features, (-1, shape[2]))
        vmask = ops.reshape(valid_mask, (-1,))
        n = ops.shape(flat)[0]
        rank = ops.cumsum(ops.cast(vmask, "int32")) - 1
        target = ops.where(vmask, rank, n)
        buffer = ops.zeros((n + 1, shape[2]), dtype=flat.dtype)
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
        # Embed text (multimodal slots replaced by pad), then scatter projected
        # vision and audio soft tokens into their placeholder positions. Returns
        # the fused embeddings and the vision-token mask (for blockwise masking).
        lm = self.language_model
        is_image = input_ids == self.image_token_id
        is_video = input_ids == self.video_token_id
        is_audio = input_ids == self.audio_token_id
        is_vision = ops.logical_or(is_image, is_video)
        multimodal = ops.logical_or(is_vision, is_audio)

        hidden = lm.embed_scaled(ops.where(multimodal, self.pad_token_id, input_ids))

        if pixel_values is not None:
            soft = self.vision_model(
                ops.convert_to_tensor(pixel_values),
                ops.cast(ops.convert_to_tensor(pixel_position_ids), "int32"),
            )
            soft = self.embed_vision(soft)
            features = ops.cast(ops.reshape(soft, (-1, lm.embed_dim)), hidden.dtype)
            hidden = self.scatter_soft_tokens(hidden, is_image, features)

        if input_features is not None and self.audio_tower is not None:
            audio_out, out_mask = self.audio_tower(
                ops.convert_to_tensor(input_features),
                None
                if input_features_mask is None
                else ops.cast(ops.convert_to_tensor(input_features_mask), "bool"),
            )
            audio_soft = self.embed_audio(audio_out)
            if out_mask is not None:
                features = self.compact_valid(audio_soft, ops.cast(out_mask, "bool"))
            else:
                features = ops.reshape(audio_soft, (-1, lm.embed_dim))
            hidden = self.scatter_soft_tokens(
                hidden, is_audio, ops.cast(features, hidden.dtype)
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
        rope = vision.get("rope_parameters") or {}
        return {
            "hidden_size": vision["hidden_size"],
            "num_layers": vision["num_hidden_layers"],
            "num_heads": vision["num_attention_heads"],
            "num_kv_heads": vision.get(
                "num_key_value_heads", vision["num_attention_heads"]
            ),
            "head_dim": vision.get("head_dim", 64),
            "intermediate_size": vision["intermediate_size"],
            "patch_size": vision.get("patch_size", 16),
            "position_embedding_size": vision.get("position_embedding_size", 10240),
            "pooling_kernel_size": vision.get("pooling_kernel_size", 3),
            "rope_theta": rope.get("rope_theta", 100.0),
            "eps": vision.get("rms_norm_eps", 1e-6),
            "standardize": bool(vision.get("standardize", False)),
            "use_clipped_linears": bool(vision.get("use_clipped_linears", False)),
        }

    @staticmethod
    def audio_config_from_hf(audio):
        return {
            "hidden_size": audio["hidden_size"],
            "num_layers": audio["num_hidden_layers"],
            "num_heads": audio["num_attention_heads"],
            "conv_channels": tuple(audio.get("subsampling_conv_channels", (128, 32))),
            "conv_kernel_size": audio.get("conv_kernel_size", 5),
            "chunk_size": audio.get("attention_chunk_size", 12),
            "context_left": audio.get("attention_context_left", 13),
            "context_right": audio.get("attention_context_right", 0),
            "logit_cap": audio.get("attention_logit_cap", 50.0),
            "invalid_logits": audio.get("attention_invalid_logits_value", -1e9),
            "residual_weight": audio.get("residual_weight", 0.5),
            "norm_eps": audio.get("rms_norm_eps", 1e-6),
            "output_proj_dims": audio.get("output_proj_dims", 1536),
            "use_clipped_linears": bool(audio.get("use_clipped_linears", True)),
        }

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config["text_config"]
        vision = hf_config["vision_config"]
        audio = hf_config.get("audio_config")
        return {
            "text_config": Gemma4Model.config_from_hf(hf_config),
            "vision_config": cls.vision_config_from_hf(vision),
            "audio_config": cls.audio_config_from_hf(audio) if audio else None,
            "image_token_id": hf_config.get("image_token_id", 258880),
            "video_token_id": hf_config.get("video_token_id", 258884),
            "audio_token_id": hf_config.get("audio_token_id", 258881),
            "pad_token_id": text.get("pad_token_id", 0),
            "use_bidirectional_vision": text.get("use_bidirectional_attention")
            == "vision",
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma4_multimodal_hf_to_keras import (
            transfer_gemma4_multimodal_weights,
        )

        transfer_gemma4_multimodal_weights(keras_model, hf_state_dict)

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
class Gemma4MultimodalGenerate(Gemma4MultimodalModel, BaseGeneration):
    """Gemma 4 multimodal backbone + a (tied) LM head with fast ``.generate()``.

    The multimodal prefill fuses image and audio soft tokens and applies the
    blockwise vision mask; decoding afterwards is text-only, so it reuses the
    per-layer sliding / global K/V cache geometry of the text decoder. Pass
    ``pixel_values`` / ``pixel_position_ids`` / ``input_features`` /
    ``input_features_mask`` as keyword prefill inputs to ``generate``.
    """

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
        for i, layer in enumerate(lm.decoder_layers):
            if lm.is_sliding(i):
                hidden, (k, v) = layer(
                    hidden, cos_l, sin_l, attention_mask=sliding_mask, use_cache=True
                )
            else:
                hidden, (k, v) = layer(
                    hidden, cos_g, sin_g, attention_mask=full_mask, use_cache=True
                )
            nkv = int(k.shape[1])
            hd = int(k.shape[3])
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
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
        h = lm.embed_scaled(ops.cast(token_ids, "int32"))
        new_caches = []
        for i, layer in enumerate(lm.decoder_layers):
            if lm.is_sliding(i):
                h, ck, cv = layer.decode_step(
                    h, cos_l, sin_l, cache[i][:, 0], cache[i][:, 1], pos, sliding_km
                )
            else:
                h, ck, cv = layer.decode_step(
                    h, cos_g, sin_g, cache[i][:, 0], cache[i][:, 1], pos, full_km
                )
            new_caches.append(ops.stack([ck, cv], axis=1))
        logits = self.project(lm.final_norm(h))[:, 0, :]
        return logits, tuple(new_caches)
