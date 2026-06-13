import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .config import JANUS_CONFIG, JANUS_WEIGHTS_URLS
from .janus_layers import (
    JanusTextDecoderLayer,
    JanusTextRMSNorm,
    JanusVisionLayer,
)

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class JanusVisionModel(layers.Layer):
    """SigLIP vision tower: biased conv patch embed + learned position
    embeddings -> pre-LN encoder blocks (exact gelu) -> final LayerNorm.

    Args:
        embed_dim: Vision hidden width.
        mlp_dim: Vision MLP hidden width.
        num_layers: Number of encoder blocks.
        num_heads: Attention heads.
        image_size: Square input size in pixels (384).
        patch_size: Patch size in pixels (16).
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
        image_size=384,
        patch_size=16,
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
            JanusVisionLayer(
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
        x = x + self.position_embedding(ops.arange(self.num_positions))[None]
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
class JanusModel(SubclassedBaseModel):
    """Janus-Pro multimodal understanding backbone: SigLIP-style tower +
    depth-2 GELU aligner + Llama decoder.

    The 384px image becomes 576 patch tokens, aligned to the text width and
    scattered into the ``image_token_id`` (``<image_placeholder>``) slots of
    the decoder input. The VQ-VAE image-generation stack of the released
    checkpoints (``vqmodel`` + generation heads) is not ported — this is the
    understanding (image+text -> text) path only. Returns raw features; use
    :class:`JanusGenerate` for logits / text.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: SwiGLU hidden width per text layer.
        num_layers: Number of text decoder blocks.
        num_heads: Query heads per text layer.
        num_kv_heads: Key/value heads per text layer.
        head_dim: Text per-head dim.
        norm_eps: Text RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        tie_embeddings: Whether :class:`JanusGenerate` ties the LM head.
        vision_embed_dim / vision_mlp_dim / vision_num_layers /
        vision_num_heads: SigLIP tower dims.
        image_size / patch_size: Vision input geometry (384 / 16).
        vision_norm_eps: Vision LayerNorm epsilon.
        image_token_id: ``<image_placeholder>`` id (100015).
    """

    HF_MODEL_TYPE = "janus"
    BASE_MODEL_CONFIG = JANUS_CONFIG
    BASE_WEIGHT_CONFIG = JANUS_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=102400,
        embed_dim=2048,
        mlp_dim=5632,
        num_layers=24,
        num_heads=16,
        num_kv_heads=16,
        head_dim=128,
        norm_eps=1e-6,
        rope_theta=10000.0,
        tie_embeddings=False,
        vision_embed_dim=1024,
        vision_mlp_dim=4096,
        vision_num_layers=24,
        vision_num_heads=16,
        image_size=384,
        patch_size=16,
        vision_norm_eps=1e-6,
        image_token_id=100581,
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
        self.vision_norm_eps = vision_norm_eps
        self.image_token_id = image_token_id

        self.vision_model = JanusVisionModel(
            vision_embed_dim,
            vision_mlp_dim,
            vision_num_layers,
            vision_num_heads,
            image_size,
            patch_size,
            vision_norm_eps,
            name="vision_model",
        )
        self.aligner_fc1 = layers.Dense(embed_dim, name="aligner_fc1")
        self.aligner_hidden = layers.Dense(embed_dim, name="aligner_hidden")
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            JanusTextDecoderLayer(
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
        self.final_norm = JanusTextRMSNorm(eps=norm_eps, name="final_norm")

    def get_image_features(self, pixel_values):
        # Janus aligner (depth 2): hidden(gelu(fc1(x))).
        features = self.vision_model(pixel_values)
        return self.aligner_hidden(
            ops.gelu(self.aligner_fc1(features), approximate=False)
        )

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
        inputs_embeds = self.token_embedding(input_ids)
        if pixel_values is not None:
            image_embeds = ops.reshape(
                self.get_image_features(pixel_values), (-1, self.embed_dim)
            )
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
        seq = int(input_ids.shape[1])
        hidden, position_ids = self.prepare_inputs(
            input_ids, inputs.get("pixel_values"), inputs.get("attention_mask")
        )
        cos, sin = self.rope_tables(position_ids)
        attn_mask = self.causal_mask(seq, inputs.get("attention_mask"))
        for layer in self.decoder_layers:
            hidden = layer(hidden, cos, sin, attention_mask=attn_mask)
        return self.final_norm(hidden)

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
            "num_kv_heads": text.get(
                "num_key_value_heads", text["num_attention_heads"]
            ),
            "head_dim": text.get("head_dim"),
            "norm_eps": text.get("rms_norm_eps", 1e-6),
            "rope_theta": text.get("rope_theta", 10000.0),
            "tie_embeddings": bool(text.get("tie_word_embeddings") or False),
            "vision_embed_dim": vision["hidden_size"],
            "vision_mlp_dim": vision["intermediate_size"],
            "vision_num_layers": vision["num_hidden_layers"],
            "vision_num_heads": vision["num_attention_heads"],
            "image_size": vision.get("image_size", 384),
            "patch_size": vision.get("patch_size", 16),
            "vision_norm_eps": vision.get("layer_norm_eps", 1e-6),
            "image_token_id": hf_config.get("image_token_id", 100581),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_janus_hf_to_keras import transfer_janus_weights

        transfer_janus_weights(keras_model, hf_state_dict)

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
                "vision_norm_eps": self.vision_norm_eps,
                "image_token_id": self.image_token_id,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class JanusGenerate(JanusModel, BaseGeneration):
    """DeepSeek-VL with an LM head + fast ``.generate()`` (image+text -> text).

    Adds a bias-free ``lm_head`` on top of :class:`JanusModel`. ``call``
    returns both ``logits`` and ``last_hidden_state``. Fast generation comes
    from :class:`~kerasformers.base.BaseGeneration`'s multimodal path:
    ``build_cache`` runs the vision tower + aligner + fused prefill ONCE
    (consuming ``pixel_values``), then ``call_with_cache`` does text-only
    decode:

        gen.generate(input_ids, pixel_values=...)
    """

    # DeepSeek <｜end▁of▁sentence｜> stop id. Explicit generate() args override.
    eos_token_id = (100001,)

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
        # Multimodal prefill into a fixed (B, num_layers, 2, nkv, max_len, hd)
        # cache. Returns (cache, last-token logits).
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        hidden, position_ids = self.prepare_inputs(
            token_ids, pixel_values, padding_mask
        )
        cos, sin = self.rope_tables(position_ids)
        causal = self.causal_mask(prompt_len, padding_mask)
        layer_caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden, cos, sin, attention_mask=causal, use_cache=True
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
        # Text-only decode step at position ``cache_update_index``.
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[:, i, 0], cache[:, i, 1], pos, key_mask
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
