import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .config import GEMMA3_CONFIG, GEMMA3_WEIGHTS_URLS
from .gemma3_layers import Gemma3DecoderLayer, Gemma3RMSNorm, Gemma3VisionLayer

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma3VisionModel(layers.Layer):
    """SigLIP vision tower: biased conv patch embed + learned position
    embeddings -> pre-LN encoder blocks -> final LayerNorm.

    Args:
        embed_dim: Vision hidden width.
        mlp_dim: Vision MLP hidden width.
        num_layers: Number of encoder blocks.
        num_heads: Attention heads.
        image_size: Square input size in pixels (896).
        patch_size: Patch size in pixels (14).
        norm_eps: LayerNorm epsilon.

    Call args:
        pixel_values: ``(num_images, H, W, 3)`` (or channels-first).

    Returns:
        ``(num_images, num_patches, embed_dim)``.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_layers,
        num_heads,
        image_size=896,
        patch_size=14,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.norm_eps = norm_eps
        self.num_positions = (image_size // patch_size) ** 2

        self.patch_embed = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            data_format="channels_last",
            name="patch_embed",
        )
        self.position_embedding = layers.Embedding(
            self.num_positions, embed_dim, name="position_embedding"
        )
        self.blocks = [
            Gemma3VisionLayer(
                embed_dim, mlp_dim, num_heads, norm_eps, name=f"blocks_{i}"
            )
            for i in range(num_layers)
        ]
        self.post_layernorm = layers.LayerNormalization(
            epsilon=norm_eps, name="post_layernorm"
        )

    def call(self, pixel_values):
        if (
            pixel_values.shape[1] is not None
            and int(pixel_values.shape[1]) == 3
            and (pixel_values.shape[-1] is None or int(pixel_values.shape[-1]) != 3)
        ):
            pixel_values = ops.transpose(pixel_values, (0, 2, 3, 1))
        x = self.patch_embed(pixel_values)
        b = ops.shape(x)[0]
        x = ops.reshape(x, (b, -1, self.embed_dim))
        positions = ops.arange(self.num_positions)
        x = x + self.position_embedding(positions)[None]
        for block in self.blocks:
            x = block(x)
        return self.post_layernorm(x)

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
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma3MultiModalProjector(layers.Layer):
    """Gemma 3 vision projector: 4x4 average pool -> soft-token RMS norm ->
    matmul with the learned ``(vision_dim, text_dim)`` projection matrix.

    Args:
        vision_dim: Vision hidden width.
        text_dim: Text decoder hidden width.
        patches_per_image: Vision patch-grid side (64 for 896/14).
        tokens_per_side: Output token-grid side (16 for 256 tokens).
        norm_eps: Epsilon of the soft-token norm.
    """

    def __init__(
        self,
        vision_dim,
        text_dim,
        patches_per_image=64,
        tokens_per_side=16,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.patches_per_image = patches_per_image
        self.tokens_per_side = tokens_per_side
        self.norm_eps = norm_eps
        self.kernel_size = patches_per_image // tokens_per_side
        self.mm_soft_emb_norm = Gemma3RMSNorm(eps=norm_eps, name="mm_soft_emb_norm")

    def build(self, input_shape):
        self.mm_input_projection_weight = self.add_weight(
            name="mm_input_projection_weight",
            shape=(self.vision_dim, self.text_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, vision_outputs):
        b = ops.shape(vision_outputs)[0]
        p, k = self.patches_per_image, self.kernel_size
        x = ops.reshape(vision_outputs, (b, p, p, self.vision_dim))
        x = ops.reshape(x, (b, p // k, k, p // k, k, self.vision_dim))
        x = ops.mean(x, axis=(2, 4))  # 4x4 average pool
        x = ops.reshape(x, (b, (p // k) * (p // k), self.vision_dim))
        x = self.mm_soft_emb_norm(x)
        return ops.matmul(x, ops.cast(self.mm_input_projection_weight, x.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_dim": self.vision_dim,
                "text_dim": self.text_dim,
                "patches_per_image": self.patches_per_image,
                "tokens_per_side": self.tokens_per_side,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma3Model(SubclassedBaseModel):
    """Gemma 3 backbone — text decoder, optionally with the SigLIP tower and
    average-pool projector (4B/12B/27B; the 1B is text-only).

    The decoder uses Gemma 3's recipe: scaled embeddings, ``(1 + w)``
    RMSNorms, per-head QK norms, the four-norm sandwich, a 5:1
    sliding-to-global layer pattern (``sliding_window_pattern``), and
    *dual rotary bases* — sliding layers use ``rope_local_theta`` (10k,
    unscaled), global layers ``rope_theta`` (1M) with an optional linear
    ``rope_scaling_factor`` (8 on 4B+). Projected image embeddings replace
    the ``image_token_id`` placeholder slots, and image-token groups attend
    *bidirectionally* (OR-ed into both the causal and sliding masks).
    Returns raw features; use :class:`Gemma3Generate` for logits / text.

    Output dict:

    .. code-block:: python

        out = model({
            "input_ids": ...,     # (B, L) int, <image_soft_token> placeholders
            "pixel_values": ...,  # (num_images, 896, 896, 3)
        })
        out["last_hidden_state"]

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: GeGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer.
        head_dim: Per-head dim.
        query_pre_attn_scalar: Attention scaling denominator.
        sliding_window: Window of the sliding layers.
        sliding_window_pattern: Every ``pattern``-th layer is global (6).
        norm_eps: RMSNorm epsilon.
        rope_theta: Global-layer rotary base (1e6).
        rope_local_theta: Sliding-layer rotary base (1e4).
        rope_scaling_factor: Linear factor dividing the global-layer inverse
            frequencies (``None`` disables; 8.0 on the multimodal sizes).
        tie_embeddings: Whether :class:`Gemma3Generate` ties the LM head.
        vision_embed_dim / vision_mlp_dim / vision_num_layers /
        vision_num_heads: SigLIP tower dims (``vision_num_layers=0`` builds
            the text-only 1B).
        image_size / patch_size: Vision input geometry (896 / 14).
        vision_norm_eps: Vision LayerNorm epsilon.
        mm_tokens_per_image: Image tokens after pooling (256).
        image_token_id: ``<image_soft_token>`` placeholder id (262144).
    """

    HF_MODEL_TYPE = ("gemma3", "gemma3_text")
    BASE_MODEL_CONFIG = GEMMA3_CONFIG
    BASE_WEIGHT_CONFIG = GEMMA3_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=262144,
        embed_dim=1152,
        mlp_dim=6912,
        num_layers=26,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        query_pre_attn_scalar=256.0,
        sliding_window=512,
        sliding_window_pattern=6,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_local_theta=10000.0,
        rope_scaling_factor=None,
        tie_embeddings=True,
        vision_embed_dim=1152,
        vision_mlp_dim=4304,
        vision_num_layers=0,
        vision_num_heads=16,
        image_size=896,
        patch_size=14,
        vision_norm_eps=1e-6,
        mm_tokens_per_image=256,
        image_token_id=262144,
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
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.rope_local_theta = rope_local_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.tie_embeddings = tie_embeddings
        self.vision_embed_dim = vision_embed_dim
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_norm_eps = vision_norm_eps
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_id = image_token_id

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            Gemma3DecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                query_pre_attn_scalar,
                norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = Gemma3RMSNorm(eps=norm_eps, name="final_norm")
        if vision_num_layers:
            self.vision_tower = Gemma3VisionModel(
                vision_embed_dim,
                vision_mlp_dim,
                vision_num_layers,
                vision_num_heads,
                image_size,
                patch_size,
                vision_norm_eps,
                name="vision_tower",
            )
            self.multi_modal_projector = Gemma3MultiModalProjector(
                vision_embed_dim,
                embed_dim,
                image_size // patch_size,
                int(mm_tokens_per_image**0.5),
                vision_norm_eps,
                name="multi_modal_projector",
            )
        else:
            self.vision_tower = None
            self.multi_modal_projector = None

    def is_sliding(self, layer_idx):
        return bool((layer_idx + 1) % self.sliding_window_pattern)

    def embed_scaled(self, input_ids):
        return self.token_embedding(input_ids) * ops.cast(
            self.embed_dim**0.5, self.compute_dtype
        )

    def rope_tables(self, position_ids, local):
        hd = self.head_dim
        theta = self.rope_local_theta if local else self.rope_theta
        inv_freq = 1.0 / ops.power(theta, ops.arange(0, hd, 2, dtype="float32") / hd)
        if not local and self.rope_scaling_factor is not None:
            inv_freq = inv_freq / self.rope_scaling_factor
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        return (
            ops.cast(ops.cos(emb), self.compute_dtype),
            ops.cast(ops.sin(emb), self.compute_dtype),
        )

    def image_groups(self, input_ids):
        # Consecutive runs of image placeholder tokens get a group id (>= 0);
        # everything else -1. Image groups attend bidirectionally.
        is_image = ops.cast(input_ids == self.image_token_id, "int32")
        prev = ops.concatenate(
            [ops.zeros_like(is_image[:, :1]), is_image[:, :-1]], axis=1
        )
        new_start = is_image * (1 - prev)
        groups = ops.cumsum(new_start, axis=1) - 1
        return ops.where(is_image > 0, groups, -1)

    def build_masks(self, input_ids, attention_mask=None):
        # (full, sliding) additive masks with the image-bidirectional overlay.
        seq = int(input_ids.shape[1])
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = (ki <= qi)[None]
        in_window = (ki > qi - self.sliding_window)[None]
        groups = self.image_groups(ops.cast(ops.convert_to_tensor(input_ids), "int32"))
        same_image = ops.logical_and(
            groups[:, :, None] == groups[:, None, :], (groups >= 0)[:, :, None]
        )
        full_keep = ops.logical_or(causal, same_image)
        sliding_keep = ops.logical_or(ops.logical_and(causal, in_window), same_image)
        full = ops.cast(ops.where(full_keep, 0.0, MASK_NEG), "float32")[:, None]
        sliding = ops.cast(ops.where(sliding_keep, 0.0, MASK_NEG), "float32")[:, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            pad = (1.0 - am)[:, None, None, :] * MASK_NEG
            full = full + pad
            sliding = sliding + pad
        return full, sliding

    def prepare_inputs(self, input_ids, pixel_values, attention_mask):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        inputs_embeds = self.embed_scaled(input_ids)
        if pixel_values is not None and self.vision_tower is not None:
            features = self.vision_tower(pixel_values)
            image_embeds = self.multi_modal_projector(features)
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
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        inputs_embeds, position_ids = self.prepare_inputs(
            input_ids, inputs.get("pixel_values"), inputs.get("attention_mask")
        )
        cos_l, sin_l = self.rope_tables(position_ids, local=True)
        cos_g, sin_g = self.rope_tables(position_ids, local=False)
        full_mask, sliding_mask = self.build_masks(
            input_ids, inputs.get("attention_mask")
        )
        hidden = inputs_embeds
        for i, layer in enumerate(self.decoder_layers):
            if self.is_sliding(i):
                hidden = layer(hidden, cos_l, sin_l, attention_mask=sliding_mask)
            else:
                hidden = layer(hidden, cos_g, sin_g, attention_mask=full_mask)
        return self.final_norm(hidden)

    def call(self, inputs):
        return {"last_hidden_state": self.forward_features(inputs)}

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config.get("text_config", hf_config)
        vision = hf_config.get("vision_config")
        rope_scaling = text.get("rope_scaling") or {}
        out = {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text.get(
                "num_key_value_heads", text["num_attention_heads"]
            ),
            "head_dim": text.get("head_dim", 256),
            "query_pre_attn_scalar": text.get("query_pre_attn_scalar", 256.0),
            "sliding_window": text.get("sliding_window", 512),
            "sliding_window_pattern": text.get("sliding_window_pattern", 6),
            "norm_eps": text.get("rms_norm_eps", 1e-6),
            "rope_theta": text.get("rope_theta", 1000000.0),
            "rope_local_theta": text.get("rope_local_base_freq", 10000.0),
            "rope_scaling_factor": rope_scaling.get("factor"),
            "tie_embeddings": text.get("tie_word_embeddings", True),
            "image_token_id": hf_config.get(
                "image_token_id", hf_config.get("image_token_index", 262144)
            ),
            "mm_tokens_per_image": hf_config.get("mm_tokens_per_image", 256),
        }
        if vision is not None:
            out.update(
                {
                    "vision_embed_dim": vision["hidden_size"],
                    "vision_mlp_dim": vision["intermediate_size"],
                    "vision_num_layers": vision["num_hidden_layers"],
                    "vision_num_heads": vision["num_attention_heads"],
                    "image_size": vision.get("image_size", 896),
                    "patch_size": vision.get("patch_size", 14),
                    "vision_norm_eps": vision.get("layer_norm_eps", 1e-6),
                }
            )
        return out

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma3_hf_to_keras import transfer_gemma3_weights

        transfer_gemma3_weights(keras_model, hf_state_dict)

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
                "query_pre_attn_scalar": self.query_pre_attn_scalar,
                "sliding_window": self.sliding_window,
                "sliding_window_pattern": self.sliding_window_pattern,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "rope_local_theta": self.rope_local_theta,
                "rope_scaling_factor": self.rope_scaling_factor,
                "tie_embeddings": self.tie_embeddings,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_norm_eps": self.vision_norm_eps,
                "mm_tokens_per_image": self.mm_tokens_per_image,
                "image_token_id": self.image_token_id,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma3Generate(Gemma3Model, BaseGeneration):
    """Gemma 3 with a (tied) LM head + fast ``.generate()`` (text or
    image+text -> text).

    The vocabulary projection is the transposed token embedding
    (``tie_embeddings``, all Gemma 3 checkpoints). Fast generation comes from
    :class:`~kerasformers.base.BaseGeneration`'s multimodal path:
    ``build_cache`` runs the vision tower + projector + fused prefill ONCE
    (consuming ``pixel_values``) with the image-bidirectional masks, then
    ``call_with_cache`` does text-only decode:

        gen.generate(input_ids, pixel_values=...)
    """

    # Gemma <eos> / <end_of_turn> stop ids. Explicit generate() args override.
    eos_token_id = (1, 106)

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

    def build_cache(self, token_ids, padding_mask, max_len, pixel_values=None):
        # Multimodal prefill with the image-bidirectional masks; each layer's
        # K/V lands in a fixed (B, num_layers, 2, nkv, max_len, hd) cache.
        batch = int(token_ids.shape[0])
        hd, nkv = self.head_dim, self.num_kv_heads
        inputs_embeds, position_ids = self.prepare_inputs(
            token_ids, pixel_values, padding_mask
        )
        cos_l, sin_l = self.rope_tables(position_ids, local=True)
        cos_g, sin_g = self.rope_tables(position_ids, local=False)
        full_mask, sliding_mask = self.build_masks(token_ids, padding_mask)
        hidden = inputs_embeds
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            if self.is_sliding(i):
                hidden, (k, v) = layer(
                    hidden, cos_l, sin_l, attention_mask=sliding_mask, use_cache=True
                )
            else:
                hidden, (k, v) = layer(
                    hidden, cos_g, sin_g, attention_mask=full_mask, use_cache=True
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
        # Text-only decode step; sliding layers see only their window.
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos_l, sin_l = self.rope_tables(positions, local=True)
        cos_g, sin_g = self.rope_tables(positions, local=False)
        ar = ops.arange(max_len)
        full_km = ops.cast(ops.where(ar <= pos, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        sliding_km = ops.cast(
            ops.where(
                ops.logical_and(ar <= pos, ar > pos - self.sliding_window),
                0.0,
                MASK_NEG,
            ),
            "float32",
        )[None, None, None, :]
        h = self.embed_scaled(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            if self.is_sliding(i):
                h, ck, cv = layer.decode_step(
                    h, cos_l, sin_l, cache[:, i, 0], cache[:, i, 1], pos, sliding_km
                )
            else:
                h, ck, cv = layer.decode_step(
                    h, cos_g, sin_g, cache[:, i, 0], cache[:, i, 1], pos, full_km
                )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
