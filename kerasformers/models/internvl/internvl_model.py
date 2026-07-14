import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .internvl_config import INTERNVL_CONFIG, INTERNVL_WEIGHTS_URLS
from .internvl_layers import (
    InternVLDecoderLayer,
    InternVLMultiModalProjector,
    InternVLRMSNorm,
    InternVLVisionEmbeddings,
    InternVLVisionLayer,
    make_vision_norm,
)

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLVisionModel(layers.Layer):
    """InternViT vision tower: conv patch embed + CLS/pos embeddings ->
    layer-scaled pre-norm blocks (-> optional final LayerNorm).

    The 300M tower (1B-14B checkpoints) uses LayerNorm blocks with biased
    attention; the 6B tower (38B/78B) uses RMSNorm blocks with bias-free
    attention and full-width QK RMS-norm. With ``use_mean_pooling`` (every
    InternVL3 checkpoint) the final norm is the identity, matching HF.

    Args:
        embed_dim: Vision hidden width.
        mlp_dim: Vision MLP hidden width.
        num_layers: Number of vision blocks.
        num_heads: Vision attention heads.
        image_size: Pretrained square input size in pixels.
        patch_size: Patch size in pixels.
        attention_bias: Whether vision q/k/v carry a bias.
        qk_norm: Whether vision attention RMS-normalizes full-width q/k.
        norm_type: ``"layer_norm"`` (300M) or ``"rms_norm"`` (6B).
        norm_eps: Norm epsilon.
        layer_scale_init: Initial layer-scale (overwritten by checkpoints).
        use_mean_pooling: When ``True`` the final norm is skipped.

    Call args:
        pixel_values: ``(num_tiles, H, W, 3)`` (or channels-first).

    Returns:
        ``(num_tiles, num_patches + 1, embed_dim)`` token sequence (CLS first).
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_layers,
        num_heads,
        image_size=448,
        patch_size=14,
        attention_bias=True,
        qk_norm=False,
        norm_type="layer_norm",
        norm_eps=1e-6,
        layer_scale_init=0.1,
        use_mean_pooling=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.attention_bias = attention_bias
        self.qk_norm = qk_norm
        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.layer_scale_init = layer_scale_init
        self.use_mean_pooling = use_mean_pooling

        self.embeddings = InternVLVisionEmbeddings(
            embed_dim, image_size, patch_size, name="embeddings"
        )
        self.blocks = [
            InternVLVisionLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                attention_bias,
                qk_norm,
                norm_type,
                norm_eps,
                layer_scale_init,
                name=f"blocks_{i}",
            )
            for i in range(num_layers)
        ]
        self.vision_norm = (
            None
            if use_mean_pooling
            else make_vision_norm(norm_type, norm_eps, "vision_norm")
        )

    def call(self, pixel_values):
        hidden = self.embeddings(pixel_values)
        for block in self.blocks:
            hidden = block(hidden)
        if self.vision_norm is not None:
            hidden = self.vision_norm(hidden)
        return hidden

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "attention_bias": self.attention_bias,
                "qk_norm": self.qk_norm,
                "norm_type": self.norm_type,
                "norm_eps": self.norm_eps,
                "layer_scale_init": self.layer_scale_init,
                "use_mean_pooling": self.use_mean_pooling,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLTextModel(layers.Layer):
    """Qwen2-style causal decoder: ``embed -> num_layers x InternVLDecoderLayer
    -> RMSNorm``.

    The token embedding lives here (``token_embedding``); ``call`` takes the
    pre-computed multimodal-fused ``inputs_embeds`` and rotary tables, and
    threads an optional KV cache for incremental decoding.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: SwiGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.
        norm_eps: RMSNorm epsilon.

    Call args:
        inputs_embeds: ``(batch, seq, embed_dim)`` fused token + vision embeds.
        cos, sin: rotary tables ``(batch, seq, head_dim)``.
        attention_mask: additive mask broadcastable to
            ``(batch, 1, q_len, kv_len)``, or ``None``.
        past_key_values: optional list of per-layer ``(key, value)`` entries.
        use_cache: when ``True``, also return the updated per-layer cache.

    Returns:
        ``(batch, seq, embed_dim)``, or ``(hidden, new_cache)`` when
        ``use_cache``.
    """

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
            InternVLDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                self.head_dim,
                norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = InternVLRMSNorm(eps=norm_eps, name="final_norm")

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
class InternVLModel(SubclassedBaseModel):
    """InternVL3 multimodal backbone: InternViT tower + pixel-shuffle projector
    + Qwen2-style decoder.

    Tiled 448x448 images run through the vision tower; the CLS token is
    dropped, the 32x32 patch grid is pixel-shuffled (``downsample_ratio`` 0.5,
    so 4 neighbouring patches fuse channel-wise into one of 256 tokens per
    tile), projected to the text width, and scattered into the
    ``image_token_id`` (``<IMG_CONTEXT>``) placeholder slots of the decoder
    input. Standard 1D rotary positions. The forward pass runs eagerly with
    ``keras.ops``. Returns raw features; use :class:`InternVLGenerate` for
    logits / text.

    Output dict:

    .. code-block:: python

        out = model({
            "input_ids": ...,      # (B, L) int, <IMG_CONTEXT> placeholders
            "pixel_values": ...,   # (num_tiles, 448, 448, 3) image tiles
        })
        out["last_hidden_state"]   # (B, L, embed_dim)

    ``pixel_values`` is optional — text-only inputs work unchanged.

    Construction:

    >>> InternVLModel.from_weights("internvl3-1b")
    >>> InternVLModel.from_weights("hf:OpenGVLab/InternVL3-1B-hf")

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: SwiGLU hidden width per text layer.
        num_layers: Number of text decoder blocks.
        num_heads: Query heads per text layer.
        num_kv_heads: Key/value heads per text layer (GQA).
        norm_eps: Text RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        tie_embeddings: Whether :class:`InternVLGenerate` ties the LM head
            (every InternVL3-hf checkpoint materializes ``lm_head``: False).
        vision_embed_dim, vision_mlp_dim, vision_num_layers, vision_num_heads:
            InternViT tower dimensions.
        image_size: Tile size in pixels (448).
        patch_size: Vision patch size in pixels (14).
        vision_attention_bias: Whether vision q/k/v carry a bias (300M: True).
        vision_qk_norm: Vision full-width QK RMS-norm (6B tower: True).
        vision_norm_type: ``"layer_norm"`` (300M) or ``"rms_norm"`` (6B).
        vision_norm_eps: Vision norm epsilon.
        vision_layer_scale_init: Initial vision layer-scale value.
        downsample_ratio: Pixel-shuffle scale factor (0.5).
        image_token_id: ``<IMG_CONTEXT>`` placeholder id replaced by projected
            vision tokens.
    """

    HF_MODEL_TYPE = "internvl"
    BASE_MODEL_CONFIG = INTERNVL_CONFIG
    BASE_WEIGHT_CONFIG = INTERNVL_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=151674,
        embed_dim=896,
        mlp_dim=4864,
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_embeddings=False,
        vision_embed_dim=1024,
        vision_mlp_dim=4096,
        vision_num_layers=24,
        vision_num_heads=16,
        image_size=448,
        patch_size=14,
        vision_attention_bias=True,
        vision_qk_norm=False,
        vision_norm_type="layer_norm",
        vision_norm_eps=1e-6,
        vision_layer_scale_init=0.1,
        downsample_ratio=0.5,
        image_token_id=151667,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings
        self.vision_embed_dim = vision_embed_dim
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_attention_bias = vision_attention_bias
        self.vision_qk_norm = vision_qk_norm
        self.vision_norm_type = vision_norm_type
        self.vision_norm_eps = vision_norm_eps
        self.vision_layer_scale_init = vision_layer_scale_init
        self.downsample_ratio = downsample_ratio
        self.image_token_id = image_token_id
        self.projector_input_dim = vision_embed_dim * int(1 / downsample_ratio) ** 2

        self.vision_tower = InternVLVisionModel(
            embed_dim=vision_embed_dim,
            mlp_dim=vision_mlp_dim,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads,
            image_size=image_size,
            patch_size=patch_size,
            attention_bias=vision_attention_bias,
            qk_norm=vision_qk_norm,
            norm_type=vision_norm_type,
            norm_eps=vision_norm_eps,
            layer_scale_init=vision_layer_scale_init,
            name="vision_tower",
        )
        self.multi_modal_projector = InternVLMultiModalProjector(
            self.projector_input_dim, embed_dim, name="multi_modal_projector"
        )
        self.language_model = InternVLTextModel(
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

    def pixel_shuffle(self, vision_features):
        # Port of HF InternVLModel.pixel_shuffle on (B, W, H, C) feature maps:
        # fuse each (1/scale x 1/scale) patch group channel-wise.
        scale = self.downsample_ratio
        b = ops.shape(vision_features)[0]
        w = int(vision_features.shape[1])
        h = int(vision_features.shape[2])
        c = int(vision_features.shape[3])
        x = ops.reshape(vision_features, (b, w, int(h * scale), int(c / scale)))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, int(h * scale), int(w * scale), int(c / (scale**2))))
        return ops.transpose(x, (0, 2, 1, 3))

    def get_image_features(self, pixel_values):
        # Vision tower -> drop CLS -> spatial grid -> pixel shuffle -> project.
        features = self.vision_tower(pixel_values)[:, 1:, :]
        n = int(features.shape[1])
        fs = int(round(n**0.5))
        features = ops.reshape(features, (-1, fs, fs, self.vision_embed_dim))
        features = self.pixel_shuffle(features)
        features = ops.reshape(
            features, (ops.shape(features)[0], -1, self.projector_input_dim)
        )
        return self.multi_modal_projector(features)

    def rope_tables(self, position_ids):
        hd = self.head_dim
        inv_freq = 1.0 / ops.power(
            self.rope_theta, ops.arange(0, hd, 2, dtype="float32") / hd
        )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        return (
            ops.cast(ops.cos(emb), self.compute_dtype),
            ops.cast(ops.sin(emb), self.compute_dtype),
        )

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def prepare_inputs(self, input_ids, pixel_values, attention_mask):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        inputs_embeds = self.language_model.token_embedding(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values)
            image_embeds = ops.reshape(image_embeds, (-1, self.embed_dim))
            ids_flat = ops.convert_to_numpy(ops.reshape(input_ids, (-1,))).tolist()
            idx = [j for j, v in enumerate(ids_flat) if v == self.image_token_id]
            embeds_flat = ops.reshape(inputs_embeds, (batch * seq, self.embed_dim))
            embeds_flat = ops.scatter_update(
                embeds_flat,
                ops.reshape(ops.convert_to_tensor(idx, dtype="int32"), (-1, 1)),
                ops.cast(image_embeds, embeds_flat.dtype),
            )
            inputs_embeds = ops.reshape(embeds_flat, (batch, seq, self.embed_dim))
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        return inputs_embeds, position_ids

    def forward_features(self, inputs):
        if not isinstance(inputs, dict):
            raise ValueError(f"{type(self).__name__} expects a dict of inputs.")
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        seq = int(input_ids.shape[1])
        inputs_embeds, position_ids = self.prepare_inputs(
            input_ids, inputs.get("pixel_values"), inputs.get("attention_mask")
        )
        cos, sin = self.rope_tables(position_ids)
        attn_mask = self.causal_mask(seq, inputs.get("attention_mask"))
        return self.language_model(inputs_embeds, cos, sin, attention_mask=attn_mask)

    def call(self, inputs):
        return {"last_hidden_state": self.forward_features(inputs)}

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config["text_config"]
        vision = hf_config["vision_config"]
        return {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text["num_key_value_heads"],
            "norm_eps": text.get("rms_norm_eps", 1e-6),
            "rope_theta": text.get("rope_theta", 1000000.0),
            "tie_embeddings": bool(text.get("tie_word_embeddings") or False),
            "vision_embed_dim": vision["hidden_size"],
            "vision_mlp_dim": vision["intermediate_size"],
            "vision_num_layers": vision["num_hidden_layers"],
            "vision_num_heads": vision["num_attention_heads"],
            "image_size": (
                vision["image_size"][0]
                if isinstance(vision.get("image_size"), (list, tuple))
                else vision.get("image_size", 448)
            ),
            "patch_size": (
                vision["patch_size"][0]
                if isinstance(vision.get("patch_size"), (list, tuple))
                else vision.get("patch_size", 14)
            ),
            "vision_attention_bias": vision.get("attention_bias", True),
            "vision_qk_norm": vision.get("use_qk_norm", False),
            "vision_norm_type": vision.get("norm_type", "layer_norm"),
            "vision_norm_eps": vision.get("layer_norm_eps", 1e-6),
            "vision_layer_scale_init": vision.get("layer_scale_init_value", 0.1),
            "downsample_ratio": hf_config.get("downsample_ratio", 0.5),
            "image_token_id": hf_config.get("image_token_id", 151667),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_internvl_hf_to_keras import transfer_internvl_weights

        transfer_internvl_weights(keras_model, hf_state_dict)

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
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "tie_embeddings": self.tie_embeddings,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_attention_bias": self.vision_attention_bias,
                "vision_qk_norm": self.vision_qk_norm,
                "vision_norm_type": self.vision_norm_type,
                "vision_norm_eps": self.vision_norm_eps,
                "vision_layer_scale_init": self.vision_layer_scale_init,
                "downsample_ratio": self.downsample_ratio,
                "image_token_id": self.image_token_id,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class InternVLGenerate(InternVLModel, BaseGeneration):
    """InternVL3 with an LM head + fast ``.generate()`` (image+text -> text).

    Adds a vocabulary projection on top of :class:`InternVLModel` (a separate
    bias-free ``lm_head`` when ``tie_embeddings`` is ``False`` — every
    InternVL3-hf checkpoint — else the tied token embedding). ``call`` returns
    both ``logits`` and ``last_hidden_state``. Fast generation comes from
    :class:`~kerasformers.base.BaseGeneration`'s multimodal path:
    ``build_cache`` runs the vision tower + projector + fused prefill ONCE
    (consuming ``pixel_values``) into a fixed KV cache, then
    ``call_with_cache`` does text-only decode. Pass pixels exactly as for
    :class:`InternVLModel`:

        gen.generate(input_ids, pixel_values=...)
    """

    # Qwen's <|im_end|> stop id. Explicit generate() args override this.
    eos_token_id = (151645,)

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
        return ops.matmul(
            hidden, ops.transpose(self.language_model.token_embedding.embeddings)
        )

    def call(self, inputs):
        hidden = self.forward_features(inputs)
        return {"logits": self.project(hidden), "last_hidden_state": hidden}

    def build_cache(self, token_ids, padding_mask, max_len, pixel_values=None):
        # Multimodal prefill: vision tower + projector + placeholder scatter,
        # then the text decoder writes each layer's K/V into a fixed
        # (B, num_layers, 2, num_kv_heads, max_len, head_dim) cache.
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        nkv = self.language_model.num_kv_heads
        hd = self.language_model.head_dim
        inputs_embeds, position_ids = self.prepare_inputs(
            token_ids, pixel_values, padding_mask
        )
        cos, sin = self.rope_tables(position_ids)
        causal = self.causal_mask(prompt_len, padding_mask)
        hidden, kv = self.language_model(
            inputs_embeds, cos, sin, attention_mask=causal, use_cache=True
        )
        layer_caches = []
        for k, v in kv:
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(hidden[:, -1, :])  # language_model already final-normed
        return cache, logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        # Text-only decode step at position ``cache_update_index``.
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.language_model.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.language_model.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[:, i, 0], cache[:, i, 1], pos, key_mask
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.language_model.final_norm(h))[:, 0, :]
        return logits, cache
