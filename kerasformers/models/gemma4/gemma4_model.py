import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel

from .gemma4_audio_layers import (
    Gemma4AudioLayer,
    Gemma4AudioRelPositionalEncoding,
    Gemma4AudioSubSampleConvProjection,
)
from .gemma4_config import (
    GEMMA4_CONFIG,
    GEMMA4_GENERATE_CONFIG,
    GEMMA4_GENERATE_WEIGHTS_URLS,
    GEMMA4_MULTIMODAL_CONFIG,
    GEMMA4_MULTIMODAL_WEIGHTS_URLS,
    GEMMA4_WEIGHTS_URLS,
)
from .gemma4_layers import Gemma4DecoderLayer, Gemma4RMSNorm
from .gemma4_vision_layers import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionEncoderLayer,
    Gemma4VisionPatchEmbedder,
    Gemma4VisionPooler,
    Gemma4VisionRotaryEmbedding,
)

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4Model(SubclassedBaseModel):
    """Gemma 4 text decoder backbone (no LM head).

    Gemma's scaled embeddings and ``(1 + w)`` norms with Gemma 4's per-layer
    attention geometry: sliding layers (5:1 pattern) use ``head_dim`` 256
    with full default rope (theta 1e4); global layers use
    ``global_head_dim`` 512 with few K/V heads, ``K = V`` attention (no value
    projection: the value is the weightlessly-normed key projection), and
    "proportional" *partial* rotary (the first quarter of the head, theta
    1e6). Attention scores are unscaled; per-head q/k norms carry the scale.
    Feed-forwards are GeGLU; on the 26B-A4B a parallel 128-expert top-8
    branch (per-expert-scaled router) is added. Each layer's output is
    multiplied by a learned ``layer_scalar``. This is the text tower only; the
    vision and audio towers of the multimodal checkpoints live in
    :class:`Gemma4MultimodalModel` (also in this module; loading a multimodal
    checkpoint here transfers just its ``model.*`` text
    weights). E2B/E4B variants (per-layer inputs, shared KV) are not
    supported. Returns raw features; use :class:`Gemma4Generate`.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: Dense GeGLU hidden width per layer.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: K/V heads on sliding layers.
        num_global_kv_heads: K/V heads on global layers.
        head_dim: Sliding-layer per-head dim (256).
        global_head_dim: Global-layer per-head dim (512).
        k_eq_v: Global layers reuse the key projection as the value.
        enable_moe: Whether layers carry the parallel expert branch.
        num_experts / num_experts_per_tok / moe_mlp_dim: MoE parameters.
        sliding_window: Window of the sliding layers.
        sliding_window_pattern: Every ``pattern``-th layer is global (6).
        partial_rotary_factor: Fraction of the global head that is rotated.
        final_logit_softcapping: LM-head tanh softcap (30.0).
        norm_eps: RMSNorm epsilon.
        rope_theta: Global-layer rotary base (1e6).
        rope_local_theta: Sliding-layer rotary base (1e4).
        tie_embeddings: Whether :class:`Gemma4Generate` ties the LM head.
    """

    HF_MODEL_TYPE = ("gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text")
    BASE_MODEL_CONFIG = GEMMA4_CONFIG
    BASE_WEIGHT_CONFIG = GEMMA4_WEIGHTS_URLS
    default_load_dtype = "bfloat16"  # Google ships gemma-4 in bf16

    def __init__(
        self,
        vocab_size=262144,
        embed_dim=3840,
        mlp_dim=15360,
        num_layers=48,
        num_heads=16,
        num_kv_heads=8,
        num_global_kv_heads=1,
        head_dim=256,
        global_head_dim=512,
        k_eq_v=True,
        enable_moe=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_mlp_dim=0,
        sliding_window=1024,
        sliding_window_pattern=6,
        partial_rotary_factor=0.25,
        final_logit_softcapping=30.0,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_local_theta=10000.0,
        tie_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_global_kv_heads = num_global_kv_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.k_eq_v = k_eq_v
        self.enable_moe = enable_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.partial_rotary_factor = partial_rotary_factor
        self.final_logit_softcapping = final_logit_softcapping
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.rope_local_theta = rope_local_theta
        self.tie_embeddings = tie_embeddings
        self.global_rot_dim = 2 * int(partial_rotary_factor * global_head_dim // 2)

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = []
        for i in range(num_layers):
            sliding = self.is_sliding(i)
            self.decoder_layers.append(
                Gemma4DecoderLayer(
                    embed_dim,
                    mlp_dim,
                    num_heads,
                    num_kv_heads if sliding else num_global_kv_heads,
                    head_dim if sliding else global_head_dim,
                    k_eq_v=(not sliding) and k_eq_v,
                    is_moe=enable_moe,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    moe_mlp_dim=moe_mlp_dim,
                    norm_eps=norm_eps,
                    name=f"decoder_layer_{i}",
                )
            )
        self.final_norm = Gemma4RMSNorm(eps=norm_eps, name="final_norm")

    def is_sliding(self, layer_idx):
        return bool((layer_idx + 1) % self.sliding_window_pattern)

    def embed_scaled(self, input_ids):
        return self.token_embedding(input_ids) * ops.cast(
            self.embed_dim**0.5, self.compute_dtype
        )

    def rope_tables(self, position_ids, local):
        if local:
            hd, rot = self.head_dim, self.head_dim
            theta = self.rope_local_theta
        else:
            hd, rot = self.global_head_dim, self.global_rot_dim
            theta = self.rope_theta
        inv_freq = 1.0 / ops.power(theta, ops.arange(0, rot, 2, dtype="float32") / hd)
        if rot < hd:
            inv_freq = ops.concatenate(
                [inv_freq, ops.zeros(((hd - rot) // 2,), dtype="float32")], axis=0
            )
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        emb = ops.concatenate([freqs, freqs], axis=-1)
        return (
            ops.cast(ops.cos(emb), self.compute_dtype),
            ops.cast(ops.sin(emb), self.compute_dtype),
        )

    def compute_position_ids(self, attention_mask, batch, seq):
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
            return ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        return ops.broadcast_to(ops.arange(seq), (batch, seq))

    def build_masks(self, seq, attention_mask=None, block_ids=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        causal = ki <= qi
        within = ki > qi - self.sliding_window
        if block_ids is None:
            full = ops.cast(ops.where(causal, 0.0, MASK_NEG), "float32")[None, None]
            sliding_keep = ops.logical_and(causal, within)
            sliding = ops.cast(ops.where(sliding_keep, 0.0, MASK_NEG), "float32")[
                None, None
            ]
        else:
            q_grp = block_ids[:, :, None]
            kv_grp = block_ids[:, None, :]
            block = ops.logical_and(q_grp == kv_grp, q_grp >= 0)
            full_keep = ops.broadcast_to(causal, ops.shape(block))
            sliding_keep = ops.logical_and(ops.logical_or(causal, block), within)
            full = ops.cast(ops.where(full_keep, 0.0, MASK_NEG), "float32")[:, None]
            sliding = ops.cast(ops.where(sliding_keep, 0.0, MASK_NEG), "float32")[
                :, None
            ]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            pad = (1.0 - am)[:, None, None, :] * MASK_NEG
            full = full + pad
            sliding = sliding + pad
        return full, sliding

    def run_layers(self, hidden, cos_l, sin_l, cos_g, sin_g, full_mask, sliding_mask):
        for i, layer in enumerate(self.decoder_layers):
            if self.is_sliding(i):
                hidden = layer(hidden, cos_l, sin_l, attention_mask=sliding_mask)
            else:
                hidden = layer(hidden, cos_g, sin_g, attention_mask=full_mask)
        return self.final_norm(hidden)

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        attention_mask = inputs.get("attention_mask")
        hidden = self.embed_scaled(input_ids)
        position_ids = self.compute_position_ids(attention_mask, batch, seq)
        cos_l, sin_l = self.rope_tables(position_ids, local=True)
        cos_g, sin_g = self.rope_tables(position_ids, local=False)
        full_mask, sliding_mask = self.build_masks(seq, attention_mask)
        hidden = self.run_layers(
            hidden, cos_l, sin_l, cos_g, sin_g, full_mask, sliding_mask
        )
        return {"last_hidden_state": hidden}

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config.get("text_config", hf_config)
        if text.get("hidden_size_per_layer_input") or text.get("num_kv_shared_layers"):
            raise NotImplementedError(
                "Gemma 4 E-variants (per-layer inputs / shared KV layers) are "
                "not supported by this port."
            )
        rope = text.get("rope_parameters") or {}
        full_rope = rope.get("full_attention") or {}
        sliding_rope = rope.get("sliding_attention") or {}
        return {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text.get(
                "num_key_value_heads", text["num_attention_heads"]
            ),
            "num_global_kv_heads": text.get("num_global_key_value_heads")
            or text.get("num_key_value_heads", 1),
            "head_dim": text.get("head_dim", 256),
            "global_head_dim": text.get("global_head_dim", 512),
            "k_eq_v": bool(text.get("attention_k_eq_v", False)),
            "enable_moe": bool(text.get("enable_moe_block", False)),
            "num_experts": text.get("num_experts") or 0,
            "num_experts_per_tok": text.get("top_k_experts") or 0,
            "moe_mlp_dim": text.get("moe_intermediate_size") or 0,
            "sliding_window": text.get("sliding_window", 1024),
            "sliding_window_pattern": text.get("sliding_window_pattern", 6),
            "partial_rotary_factor": full_rope.get("partial_rotary_factor", 0.25),
            "final_logit_softcapping": text.get("final_logit_softcapping"),
            "norm_eps": text.get("rms_norm_eps", 1e-6),
            "rope_theta": full_rope.get(
                "rope_theta", text.get("rope_theta", 1000000.0)
            ),
            "rope_local_theta": sliding_rope.get(
                "rope_theta", text.get("rope_local_base_freq", 10000.0)
            ),
            "tie_embeddings": text.get("tie_word_embeddings", True),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma4_hf_to_keras import transfer_gemma4_weights

        transfer_gemma4_weights(keras_model, hf_state_dict)

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
                "num_global_kv_heads": self.num_global_kv_heads,
                "head_dim": self.head_dim,
                "global_head_dim": self.global_head_dim,
                "k_eq_v": self.k_eq_v,
                "enable_moe": self.enable_moe,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "sliding_window": self.sliding_window,
                "sliding_window_pattern": self.sliding_window_pattern,
                "partial_rotary_factor": self.partial_rotary_factor,
                "final_logit_softcapping": self.final_logit_softcapping,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "rope_local_theta": self.rope_local_theta,
                "tie_embeddings": self.tie_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionModel(layers.Layer):
    """Gemma 4 vision encoder: patch embed, rotary transformer, spatial pool."""

    def __init__(
        self,
        hidden_size=768,
        num_layers=16,
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        intermediate_size=3072,
        patch_size=16,
        position_embedding_size=10240,
        pooling_kernel_size=3,
        rope_theta=100.0,
        eps=1e-6,
        standardize=False,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.pooling_kernel_size = pooling_kernel_size
        self.standardize = standardize
        self.patch_embedder = Gemma4VisionPatchEmbedder(
            hidden_size, patch_size, position_embedding_size, name="patch_embedder"
        )
        self.rotary_emb = Gemma4VisionRotaryEmbedding(
            head_dim, rope_theta, name="rotary_emb"
        )
        self.vlayers = [
            Gemma4VisionEncoderLayer(
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                eps,
                use_clipped_linears,
                name=f"layers_{i}",
            )
            for i in range(num_layers)
        ]
        self.pooler = Gemma4VisionPooler(hidden_size, name="pooler")

    def build(self, input_shape):
        if self.standardize:
            self.std_bias = self.add_weight(
                shape=(self.hidden_size,),
                initializer="zeros",
                trainable=False,
                name="std_bias",
            )
            self.std_scale = self.add_weight(
                shape=(self.hidden_size,),
                initializer="ones",
                trainable=False,
                name="std_scale",
            )

    def call(self, pixel_values, pixel_position_ids, attention_mask=None):
        padding = ops.all(pixel_position_ids == -1, axis=-1)
        h = self.patch_embedder(pixel_values, pixel_position_ids, padding)
        cos, sin = self.rotary_emb(pixel_position_ids)
        for layer in self.vlayers:
            h = layer(h, cos, sin, attention_mask)
        num_patches = int(ops.shape(pixel_values)[1])
        output_length = num_patches // (self.pooling_kernel_size**2)
        hidden = self.pooler(
            h, pixel_position_ids, padding, output_length=output_length
        )
        if self.standardize:
            hidden = (hidden - ops.cast(self.std_bias, "float32")) * ops.cast(
                self.std_scale, "float32"
            )
        return ops.cast(hidden, h.dtype)


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioModel(layers.Layer):
    """Gemma 4 audio encoder (USM conformer): subsample, conformer stack, projection."""

    def __init__(
        self,
        hidden_size=1024,
        num_layers=12,
        num_heads=8,
        conv_channels=(128, 32),
        conv_kernel_size=5,
        chunk_size=12,
        context_left=13,
        context_right=0,
        logit_cap=50.0,
        invalid_logits=-1e9,
        residual_weight=0.5,
        norm_eps=1e-6,
        output_proj_dims=1536,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.max_past_horizon = context_left - 1
        self.max_future_horizon = context_right
        self.context_size = chunk_size + self.max_past_horizon + self.max_future_horizon
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(
            conv_channels, hidden_size, norm_eps, name="subsample_conv_projection"
        )
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(
            hidden_size, self.context_size, name="rel_pos_enc"
        )
        self.alayers = [
            Gemma4AudioLayer(
                hidden_size,
                num_heads,
                chunk_size,
                context_left,
                context_right,
                conv_kernel_size,
                norm_eps,
                residual_weight,
                logit_cap,
                invalid_logits,
                use_clipped_linears,
                name=f"layers_{i}",
            )
            for i in range(num_layers)
        ]
        self.output_proj = layers.Dense(
            output_proj_dims, use_bias=True, name="output_proj"
        )

    def blocked_mask(self, valid_mask, seq_len):
        # valid_mask: [B, seq_len] bool (True = valid frame). Build the 4D
        # sliding-window bidirectional mask then fold it to blocked 5D
        # [B, 1, num_blocks, chunk, context].
        q = ops.arange(seq_len)[:, None]
        kv = ops.arange(seq_len)[None, :]
        dist = q - kv
        # HF sliding_window_mask_function((context_left - 1, context_right)):
        # left keeps 0 <= dist < max_past_horizon, right keeps -dist < context_right.
        window = ops.logical_and(dist >= 0, dist < self.max_past_horizon)
        window = ops.logical_or(
            window, ops.logical_and(dist < 0, -dist < self.max_future_horizon)
        )
        mask = ops.logical_and(window[None], valid_mask[:, None, :])
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        mask = ops.pad(mask, [[0, 0], [0, pad], [0, pad]])
        b = ops.shape(mask)[0]
        padded = num_blocks * self.chunk_size
        mask = ops.reshape(mask, (b, num_blocks, self.chunk_size, padded))
        mask = ops.pad(
            mask,
            [[0, 0], [0, 0], [0, 0], [self.max_past_horizon, self.max_future_horizon]],
        )
        starts = ops.arange(num_blocks) * self.chunk_size
        offsets = ops.arange(
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        kv_idx = starts[:, None] + offsets[None, :]
        gathered = ops.take_along_axis(mask, kv_idx[None, :, None, :], axis=3)
        return gathered[:, None]

    def call(self, input_features, input_features_mask=None):
        hidden, out_mask = self.subsample_conv_projection(
            input_features, input_features_mask
        )
        pos = self.rel_pos_enc.compute(hidden.dtype)
        seq_len = int(ops.shape(hidden)[1])
        mask = None
        if out_mask is not None:
            mask = self.blocked_mask(ops.cast(out_mask, "bool"), seq_len)
        for layer in self.alayers:
            hidden = layer(hidden, pos, mask)
        return self.output_proj(hidden), out_mask

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "chunk_size": self.chunk_size})
        return config


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
    :class:`Gemma4Generate`.

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
        # Vision and audio towers are optional: with neither, this is a plain
        # text generator, matching Gemma4ForConditionalGeneration on text input.
        self.vision_model = None
        self.embed_vision = None
        if self.vision_config is not None:
            self.vision_model = Gemma4VisionModel(
                **self.vision_config, name="vision_tower"
            )
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
        # Instantiate every sublayer weight with a minimal call before a weight
        # transfer. Feeds a dummy image when a vision tower is present, otherwise
        # a plain two-token text sequence.
        if self.vision_model is not None:
            patch = self.vision_config.get("patch_size", 16)
            pool = self.vision_config.get("pooling_kernel_size", 3)
            num_patches = pool * pool
            pixel_values = ops.zeros(
                (1, num_patches, 3 * patch * patch), dtype="float32"
            )
            coords = ops.stack(
                ops.meshgrid(ops.arange(pool), ops.arange(pool), indexing="xy"),
                axis=-1,
            )
            coords = ops.reshape(coords, (1, num_patches, 2))
            self(
                {
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
            )
        else:
            self({"input_ids": ops.zeros((1, 2), dtype="int32")})
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

        if pixel_values is not None and self.vision_model is not None:
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
        vision = hf_config.get("vision_config")
        audio = hf_config.get("audio_config")
        # Only the "gemma4" towers are ported; the "gemma4_unified" towers are a
        # different architecture, so those checkpoints load text-only here.
        vision_ok = bool(vision) and vision.get("model_type") == "gemma4_vision"
        audio_ok = bool(audio) and audio.get("model_type") == "gemma4_audio"
        return {
            "text_config": Gemma4Model.config_from_hf(hf_config),
            "vision_config": cls.vision_config_from_hf(vision) if vision_ok else None,
            "audio_config": cls.audio_config_from_hf(audio) if audio_ok else None,
            "image_token_id": hf_config.get("image_token_id", 258880),
            "video_token_id": hf_config.get("video_token_id", 258884),
            "audio_token_id": hf_config.get("audio_token_id", 258881),
            "pad_token_id": text.get("pad_token_id", 0),
            "use_bidirectional_vision": vision_ok
            and text.get("use_bidirectional_attention") == "vision",
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gemma4_hf_to_keras import transfer_gemma4_weights

        transfer_gemma4_weights(keras_model, hf_state_dict)

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
class Gemma4Generate(Gemma4MultimodalModel, BaseGeneration):
    """Gemma 4 backbone + a (tied) LM head with fast ``.generate()``.

    The single generation entry point, like transformers'
    ``Gemma4ForConditionalGeneration``: it drives text-only checkpoints and the
    vision / audio multimodal ones through the same API. When a vision or audio
    tower is present the prefill fuses the soft tokens and applies the blockwise
    vision mask; text-only prompts (or checkpoints built without towers) skip
    straight to the text decoder. Decoding is always text-only and reuses the
    per-layer sliding / global K/V cache geometry. Pass ``pixel_values`` /
    ``pixel_position_ids`` / ``input_features`` / ``input_features_mask`` as
    keyword prefill inputs to ``generate`` when the checkpoint has the towers.
    """

    HF_MODEL_TYPE = ("gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text")
    BASE_MODEL_CONFIG = GEMMA4_GENERATE_CONFIG
    BASE_WEIGHT_CONFIG = GEMMA4_GENERATE_WEIGHTS_URLS
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
