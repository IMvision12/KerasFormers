import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .mistral3_config import MISTRAL3_CONFIG, MISTRAL3_WEIGHTS_URLS
from .mistral3_layers import (
    Mistral3DecoderLayer,
    Mistral3MultiModalProjector,
    Mistral3RMSNorm,
    Mistral3VisionLayer,
)

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Mistral3VisionModel(layers.Layer):
    """Pixtral vision tower: conv patch embed -> RMS pre-norm -> 2D-rotary
    blocks, over a packed variable-resolution patch sequence.

    Images (padded to a common batch size) are patch-projected, each cropped
    to its own ``(h // patch, w // patch)`` grid, flattened and concatenated
    into one ``(1, total_patches, D)`` sequence. Every block applies full
    attention with 2D rotary positions (height uses the even-indexed
    frequencies, width the odd-indexed ones) and a block-diagonal mask so
    patches only attend within their own image. No final norm (Pixtral).

    Args:
        embed_dim: Vision hidden width.
        mlp_dim: Vision MLP hidden width.
        num_layers: Number of vision blocks.
        num_heads: Vision attention heads.
        image_size: Maximum image side (sizes the rotary meshgrid).
        patch_size: Patch size in pixels.
        rope_theta: Vision rotary base frequency.

    Call args:
        pixel_values: ``(num_images, H, W, 3)`` (or channels-first) padded
            batch.
        image_sizes: host list of per-image ``(height, width)`` pixel sizes.

    Returns:
        ``(total_patches, embed_dim)`` packed patch features.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_layers,
        num_heads,
        image_size=1540,
        patch_size=14,
        rope_theta=10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.head_dim = embed_dim // num_heads

        self.patch_conv = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=False,
            data_format="channels_last",
            name="patch_conv",
        )
        self.ln_pre = Mistral3RMSNorm(eps=1e-5, name="ln_pre")
        self.blocks = [
            Mistral3VisionLayer(embed_dim, mlp_dim, num_heads, name=f"blocks_{i}")
            for i in range(num_layers)
        ]

    def rope_2d_tables(self, grid_sizes):
        # Pixtral 2D rope: height positions take the even-indexed inverse
        # frequencies, width the odd-indexed; the per-patch table row is
        # indexed by h * max_side + w. Returns (1, total, head_dim) cos/sin.
        hd = self.head_dim
        max_side = self.image_size // self.patch_size
        freqs = 1.0 / ops.power(
            self.rope_theta, ops.arange(0, hd, 2, dtype="float32") / hd
        )
        h_idx = ops.arange(max_side, dtype="float32")
        freqs_h = h_idx[:, None] * freqs[::2][None, :]  # (side, hd/4)
        freqs_w = h_idx[:, None] * freqs[1::2][None, :]
        table = ops.concatenate(
            [
                ops.broadcast_to(freqs_h[:, None, :], (max_side, max_side, hd // 4)),
                ops.broadcast_to(freqs_w[None, :, :], (max_side, max_side, hd // 4)),
            ],
            axis=-1,
        )
        table = ops.reshape(table, (max_side * max_side, hd // 2))
        positions = []
        for grid_h, grid_w in grid_sizes:
            hpos = ops.repeat(ops.arange(grid_h), grid_w)
            wpos = ops.tile(ops.arange(grid_w), [grid_h])
            positions.append(hpos * max_side + wpos)
        pos = ops.concatenate(positions, axis=0)
        emb = ops.take(table, pos, axis=0)
        emb = ops.concatenate([emb, emb], axis=-1)
        return (
            ops.cast(ops.cos(emb), self.compute_dtype)[None],
            ops.cast(ops.sin(emb), self.compute_dtype)[None],
        )

    def block_mask(self, grid_sizes):
        counts = [gh * gw for gh, gw in grid_sizes]
        if len(counts) <= 1:
            return None
        seg = ops.concatenate(
            [ops.full((n,), i, dtype="int32") for i, n in enumerate(counts)], axis=0
        )
        mask = ops.where(seg[:, None] == seg[None, :], 0.0, MASK_NEG)
        return ops.cast(mask, "float32")[None, None]

    def call(self, pixel_values, image_sizes):
        if (
            pixel_values.shape[1] is not None
            and int(pixel_values.shape[1]) == 3
            and (pixel_values.shape[-1] is None or int(pixel_values.shape[-1]) != 3)
        ):
            pixel_values = ops.transpose(pixel_values, (0, 2, 3, 1))
        patches = self.patch_conv(pixel_values)  # (N, gh_max, gw_max, D)
        grid_sizes = [
            (int(h) // self.patch_size, int(w) // self.patch_size)
            for h, w in image_sizes
        ]
        pieces = []
        for i, (grid_h, grid_w) in enumerate(grid_sizes):
            pieces.append(
                ops.reshape(
                    patches[i, :grid_h, :grid_w, :], (grid_h * grid_w, self.embed_dim)
                )
            )
        hidden = ops.concatenate(pieces, axis=0)[None]  # (1, total, D)
        hidden = self.ln_pre(hidden)
        cos, sin = self.rope_2d_tables(grid_sizes)
        mask = self.block_mask(grid_sizes)
        for block in self.blocks:
            hidden = block(hidden, cos, sin, attention_mask=mask)
        return hidden[0]

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
                "rope_theta": self.rope_theta,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Mistral3TextModel(layers.Layer):
    """Mistral causal decoder: ``embed -> num_layers x Mistral3DecoderLayer ->
    RMSNorm``.

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
        head_dim: Per-head dim.
        norm_eps: RMSNorm epsilon.
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
        norm_eps=1e-5,
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
            Mistral3DecoderLayer(
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
        self.final_norm = Mistral3RMSNorm(eps=norm_eps, name="final_norm")

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
class Mistral3Model(SubclassedBaseModel):
    """Mistral 3 multimodal backbone (Mistral Small 3.1/3.2): Pixtral vision
    tower + 2x2 patch-merging projector + Mistral text decoder.

    Variable-resolution images are patch-encoded by the Pixtral tower (2D
    rotary, packed sequence), RMS-normed, spatially merged 2x2, projected to
    the text width, and scattered into the ``image_token_id`` (``[IMG]``)
    placeholder slots of the decoder input. The forward pass runs eagerly
    with ``keras.ops``. Returns raw features; use :class:`Mistral3Generate`
    for logits / text.

    Output dict:

    .. code-block:: python

        out = model({
            "input_ids": ...,     # (B, L) int, [IMG] placeholders
            "pixel_values": ...,  # (num_images, H, W, 3) padded batch
            "image_sizes": ...,   # (num_images, 2) per-image (height, width)
        })
        out["last_hidden_state"]  # (B, L, embed_dim)

    The vision keys are optional: text-only inputs work unchanged.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: SwiGLU hidden width per text layer.
        num_layers: Number of text decoder blocks.
        num_heads: Query heads per text layer.
        num_kv_heads: Key/value heads per text layer (GQA).
        head_dim: Text per-head dim.
        norm_eps: Text RMSNorm epsilon.
        rope_theta: Text rotary base frequency.
        tie_embeddings: Whether :class:`Mistral3Generate` ties the LM head.
        vision_embed_dim / vision_mlp_dim / vision_num_layers /
        vision_num_heads: Pixtral tower dimensions.
        image_size: Maximum image side in pixels (1540).
        patch_size: Vision patch size in pixels (14).
        vision_rope_theta: Vision rotary base frequency.
        spatial_merge_size: Projector patch-merge factor (2).
        projector_bias: Whether the projector linears carry biases.
        image_token_id: ``[IMG]`` placeholder id (10).
    """

    HF_MODEL_TYPE = "mistral3"
    BASE_MODEL_CONFIG = MISTRAL3_CONFIG
    BASE_WEIGHT_CONFIG = MISTRAL3_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=131072,
        embed_dim=5120,
        mlp_dim=32768,
        num_layers=40,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        norm_eps=1e-5,
        rope_theta=1000000000.0,
        tie_embeddings=False,
        vision_embed_dim=1024,
        vision_mlp_dim=4096,
        vision_num_layers=24,
        vision_num_heads=16,
        image_size=1540,
        patch_size=14,
        vision_rope_theta=10000.0,
        spatial_merge_size=2,
        projector_bias=False,
        image_token_id=10,
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
        self.rope_theta = rope_theta
        self.tie_embeddings = tie_embeddings
        self.vision_embed_dim = vision_embed_dim
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_rope_theta = vision_rope_theta
        self.spatial_merge_size = spatial_merge_size
        self.projector_bias = projector_bias
        self.image_token_id = image_token_id

        self.vision_tower = Mistral3VisionModel(
            embed_dim=vision_embed_dim,
            mlp_dim=vision_mlp_dim,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads,
            image_size=image_size,
            patch_size=patch_size,
            rope_theta=vision_rope_theta,
            name="vision_tower",
        )
        self.multi_modal_projector = Mistral3MultiModalProjector(
            vision_embed_dim,
            embed_dim,
            spatial_merge_size,
            norm_eps,
            projector_bias,
            name="multi_modal_projector",
        )
        self.language_model = Mistral3TextModel(
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

    def get_image_features(self, pixel_values, image_sizes):
        sizes = [
            (int(h), int(w))
            for h, w in ops.convert_to_numpy(
                ops.convert_to_tensor(image_sizes)
            ).tolist()
        ]
        grid_sizes = [(h // self.patch_size, w // self.patch_size) for h, w in sizes]
        features = self.vision_tower(pixel_values, image_sizes=sizes)
        return self.multi_modal_projector(features, grid_sizes=grid_sizes)

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

    def prepare_inputs(self, input_ids, pixel_values, image_sizes, attention_mask):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        inputs_embeds = self.language_model.token_embedding(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_sizes)
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
            input_ids,
            inputs.get("pixel_values"),
            inputs.get("image_sizes"),
            inputs.get("attention_mask"),
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
            "head_dim": text.get("head_dim"),
            "norm_eps": text.get("rms_norm_eps", 1e-5),
            "rope_theta": text.get("rope_theta", 1000000000.0),
            "tie_embeddings": bool(text.get("tie_word_embeddings") or False),
            "vision_embed_dim": vision["hidden_size"],
            "vision_mlp_dim": vision["intermediate_size"],
            "vision_num_layers": vision["num_hidden_layers"],
            "vision_num_heads": vision["num_attention_heads"],
            "image_size": vision.get("image_size", 1540),
            "patch_size": vision.get("patch_size", 14),
            "vision_rope_theta": vision.get("rope_theta", 10000.0),
            "spatial_merge_size": hf_config.get("spatial_merge_size", 2),
            "projector_bias": hf_config.get("multimodal_projector_bias", False),
            "image_token_id": hf_config.get(
                "image_token_id", hf_config.get("image_token_index", 10)
            ),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_mistral3_hf_to_keras import transfer_mistral3_weights

        transfer_mistral3_weights(keras_model, hf_state_dict)

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
                "rope_theta": self.rope_theta,
                "tie_embeddings": self.tie_embeddings,
                "vision_embed_dim": self.vision_embed_dim,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "vision_rope_theta": self.vision_rope_theta,
                "spatial_merge_size": self.spatial_merge_size,
                "projector_bias": self.projector_bias,
                "image_token_id": self.image_token_id,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Mistral3Generate(Mistral3Model, BaseGeneration):
    """Mistral 3 with an LM head + fast ``.generate()`` (image+text -> text).

    Adds a bias-free ``lm_head`` on top of :class:`Mistral3Model` (the
    checkpoints do not tie embeddings). ``call`` returns both ``logits`` and
    ``last_hidden_state``. Fast generation comes from
    :class:`~kerasformers.base.BaseGeneration`'s multimodal path:
    ``build_cache`` runs the vision tower + projector + fused prefill ONCE
    (consuming ``pixel_values`` / ``image_sizes``) into a fixed KV cache,
    then ``call_with_cache`` does text-only decode:

        gen.generate(input_ids, pixel_values=..., image_sizes=...)
    """

    # Mistral </s> stop id. Explicit generate() args override this.
    eos_token_id = (2,)

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

    def build_cache(
        self, token_ids, padding_mask, max_len, pixel_values=None, image_sizes=None
    ):
        # Multimodal prefill: vision tower + projector + placeholder scatter,
        # then the text decoder writes each layer's K/V into a fixed
        # (B, num_layers, 2, num_kv_heads, max_len, head_dim) cache.
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        nkv = self.language_model.num_kv_heads
        hd = self.language_model.head_dim
        inputs_embeds, position_ids = self.prepare_inputs(
            token_ids, pixel_values, image_sizes, padding_mask
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
