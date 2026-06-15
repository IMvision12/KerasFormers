import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel
from kerasformers.models.cohere2.cohere2_layers import (
    Cohere2DecoderLayer,
    Cohere2LayerNorm,
)

from .cohere2_vision_layers import Cohere2VisionProjector, Cohere2VisionTower
from .config import COHERE2_VISION_CONFIG, COHERE2_VISION_WEIGHTS_URLS

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2VisionModel(SubclassedBaseModel):
    """Cohere2-Vision (Command-A Vision) multimodal backbone.

    A SigLIP vision tower encodes each image tile; a pixel-shuffle SwiGLU
    projector maps the patch features to the text width, which are scattered
    into the ``image_token_id`` slots of a Cohere2 text decoder (parallel
    attn+MLP, NoPE full layers + sliding rope layers, ``logit_scale``).
    Returns raw features; use :class:`Cohere2VisionGenerate` for logits / text.

    Args:
        vocab_size / embed_dim / num_layers / num_heads / num_kv_heads /
        head_dim / mlp_dim: Text-decoder geometry.
        sliding_window / sliding_window_pattern / norm_eps / rope_theta /
        attention_bias / logit_scale / tie_embeddings: Text-decoder config.
        vision_embed_dim / vision_mlp_dim / vision_num_layers /
        vision_num_heads / image_size / patch_size / vision_norm_eps: Tower.
        downsample_factor / alignment_intermediate_size: Projector.
        image_token_id: Placeholder id replaced by projected patches (255036).
    """

    HF_MODEL_TYPE = "cohere2_vision"
    BASE_MODEL_CONFIG = COHERE2_VISION_CONFIG
    BASE_WEIGHT_CONFIG = COHERE2_VISION_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=256000,
        embed_dim=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mlp_dim=14336,
        sliding_window=4096,
        sliding_window_pattern=4,
        norm_eps=1e-5,
        rope_theta=50000.0,
        attention_bias=False,
        logit_scale=0.25,
        tie_embeddings=True,
        vision_embed_dim=1152,
        vision_mlp_dim=4304,
        vision_num_layers=27,
        vision_num_heads=16,
        image_size=512,
        patch_size=16,
        vision_norm_eps=1e-6,
        downsample_factor=2,
        alignment_intermediate_size=36864,
        image_token_id=255036,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.mlp_dim = mlp_dim
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.logit_scale = logit_scale
        self.tie_embeddings = tie_embeddings
        self.vision_embed_dim = vision_embed_dim
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_norm_eps = vision_norm_eps
        self.downsample_factor = downsample_factor
        self.alignment_intermediate_size = alignment_intermediate_size
        self.image_token_id = image_token_id
        self.layer_types = tuple(
            "full_attention"
            if (i + 1) % sliding_window_pattern == 0
            else "sliding_attention"
            for i in range(num_layers)
        )

        self.vision_tower = Cohere2VisionTower(
            vision_embed_dim,
            vision_mlp_dim,
            vision_num_layers,
            vision_num_heads,
            image_size,
            patch_size,
            vision_norm_eps,
            name="vision_tower",
        )
        self.projector = Cohere2VisionProjector(
            vision_embed_dim,
            embed_dim,
            downsample_factor,
            alignment_intermediate_size,
            name="projector",
        )
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Cohere2DecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                self.head_dim,
                self.layer_types[i],
                norm_eps=norm_eps,
                attention_bias=attention_bias,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Cohere2LayerNorm(eps=norm_eps, name="final_norm")

    def get_image_features(self, pixel_values):
        return self.projector(self.vision_tower(pixel_values))

    def rope_tables(self, position_ids):
        hd = self.head_dim
        inv_freq = 1.0 / ops.power(
            self.rope_theta, ops.arange(0, hd, 2, dtype="float32") / hd
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.repeat(freqs, 2, axis=-1)
        return (
            ops.cast(ops.cos(emb), self.compute_dtype),
            ops.cast(ops.sin(emb), self.compute_dtype),
        )

    def build_masks(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = ops.where(ki <= qi, 0.0, MASK_NEG)
        sliding = ops.where(
            ops.logical_and(ki <= qi, (qi - ki) < self.sliding_window), 0.0, MASK_NEG
        )
        full = ops.cast(causal, "float32")[None, None]
        slide = ops.cast(sliding, "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            pad = (1.0 - am)[:, None, None, :] * MASK_NEG
            full = full + pad
            slide = slide + pad
        return {"full_attention": full, "sliding_attention": slide}

    def prepare_inputs(self, inputs):
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        hidden = self.token_embedding(input_ids)
        if inputs.get("pixel_values") is not None:
            features = ops.reshape(
                self.get_image_features(ops.convert_to_tensor(inputs["pixel_values"])),
                (-1, self.embed_dim),
            )
            ids_flat = ops.convert_to_numpy(ops.reshape(input_ids, (-1,))).tolist()
            idx = [j for j, v in enumerate(ids_flat) if v == self.image_token_id]
            flat = ops.reshape(hidden, (batch * seq, self.embed_dim))
            flat = ops.scatter_update(
                flat,
                ops.reshape(ops.convert_to_tensor(idx, dtype="int32"), (-1, 1)),
                ops.cast(features, flat.dtype),
            )
            hidden = ops.reshape(flat, (batch, seq, self.embed_dim))
        return hidden

    def forward_features(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        hidden = self.prepare_inputs(inputs)
        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables(position_ids)
        masks = self.build_masks(seq, inputs.get("attention_mask"))
        for layer in self.decoder_layers:
            hidden = layer(hidden, cos, sin, attention_mask=masks[layer.layer_type])
        return self.final_norm(hidden)

    def call(self, inputs):
        return {"last_hidden_state": self.forward_features(inputs)}

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config["text_config"]
        vision = hf_config["vision_config"]
        trope = text.get("rope_parameters") or {}
        return {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text.get(
                "num_key_value_heads", text["num_attention_heads"]
            ),
            "head_dim": text.get("head_dim"),
            "mlp_dim": text["intermediate_size"],
            "sliding_window": text.get("sliding_window", 4096),
            "sliding_window_pattern": text.get("sliding_window_pattern", 4),
            "norm_eps": text.get("layer_norm_eps", 1e-5),
            "rope_theta": trope.get("rope_theta", text.get("rope_theta", 50000.0)),
            "attention_bias": bool(text.get("attention_bias") or False),
            "logit_scale": text.get("logit_scale", 0.25),
            "tie_embeddings": bool(text.get("tie_word_embeddings", True)),
            "vision_embed_dim": vision["hidden_size"],
            "vision_mlp_dim": vision["intermediate_size"],
            "vision_num_layers": vision["num_hidden_layers"],
            "vision_num_heads": vision["num_attention_heads"],
            "image_size": vision.get("image_size", 512),
            "patch_size": vision.get("patch_size", 16),
            "vision_norm_eps": vision.get("layer_norm_eps", 1e-6),
            "downsample_factor": hf_config.get("downsample_factor", 2),
            "alignment_intermediate_size": hf_config.get(
                "alignment_intermediate_size", 36864
            ),
            "image_token_id": hf_config.get("image_token_id", 255036),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_cohere2_vision_hf_to_keras import transfer_cohere2_vision_weights

        transfer_cohere2_vision_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "mlp_dim": self.mlp_dim,
                "sliding_window": self.sliding_window,
                "sliding_window_pattern": self.sliding_window_pattern,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "attention_bias": self.attention_bias,
                "logit_scale": self.logit_scale,
                "tie_embeddings": self.tie_embeddings,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_norm_eps": self.vision_norm_eps,
                "downsample_factor": self.downsample_factor,
                "alignment_intermediate_size": self.alignment_intermediate_size,
                "image_token_id": self.image_token_id,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2VisionGenerate(Cohere2VisionModel, BaseGeneration):
    """Cohere2-Vision (Command-A Vision) with an LM head + fast ``.generate()`` (image+text -> text).

    Adds a vocabulary projection on top of :class:`Cohere2VisionModel`: a
    bias-free ``lm_head`` when ``tie_embeddings`` is ``False``, otherwise the
    tied token embedding. **Unlike** the text-only
    :class:`~kerasformers.models.cohere2.cohere2_model.Cohere2Generate`, the
    logits are *not* multiplied by ``logit_scale`` — the HF VLM forward omits
    that scaling. ``call`` returns both ``logits`` and ``last_hidden_state``.

    Fast generation uses :class:`~kerasformers.base.BaseGeneration`'s fixed-cache
    compiled loop. :meth:`build_cache` runs the vision tower + projector + fused
    multimodal prefill once — scattering the projected patches into the
    ``image_token_id`` slots, consuming ``pixel_values`` — then
    :meth:`call_with_cache` decodes text-only over the full-length per-layer KV
    cache: the sliding-window layers enforce their window through the decode
    key-mask and the full/NoPE layers see all keys, so the loop stays
    constant-shape. ``eos_token_id`` defaults to Cohere's
    ``<|END_OF_TURN_TOKEN|>`` (255001).

    Construction mirrors :class:`Cohere2VisionModel`::

        gen = Cohere2VisionGenerate.from_weights("hf:CohereLabs/command-a-vision-07-2025")
        out = gen.generate(input_ids, pixel_values=pixels, max_new_tokens=64)
    """

    eos_token_id = (255001,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = (
            None
            if self.tie_embeddings
            else layers.Dense(self.vocab_size, use_bias=False, name="lm_head")
        )

    def project(self, hidden):
        # NB: unlike the text-only Cohere2 head, Cohere2-Vision does NOT scale
        # the logits by ``logit_scale`` (the HF VLM forward omits it).
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def call(self, inputs):
        hidden = self.forward_features(inputs)
        return {"logits": self.project(hidden), "last_hidden_state": hidden}

    def build_cache(self, token_ids, padding_mask, max_len, pixel_values=None):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        hidden = self.prepare_inputs(
            {"input_ids": token_ids, "pixel_values": pixel_values}
        )
        position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos_p, sin_p = self.rope_tables(position_ids)
        masks = self.build_masks(prompt_len, padding_mask)
        layer_caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden,
                cos_p,
                sin_p,
                attention_mask=masks[layer.layer_type],
                use_cache=True,
            )
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return cache, logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos_t, sin_t = self.rope_tables(positions)
        ar = ops.arange(max_len)
        full_mask = ops.cast(ops.where(ar <= pos, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        slide_mask = ops.cast(
            ops.where(
                ops.logical_and(ar <= pos, (pos - ar) < self.sliding_window),
                0.0,
                MASK_NEG,
            ),
            "float32",
        )[None, None, None, :]
        masks = {"full_attention": full_mask, "sliding_attention": slide_mask}
        h = self.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h,
                cos_t,
                sin_t,
                cache[:, i, 0],
                cache[:, i, 1],
                pos,
                masks[layer.layer_type],
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
