import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, SubclassedBaseModel
from kerasformers.models.deepseek_vl.deepseek_vl_layers import (
    DeepseekVLTextDecoderLayer,
    DeepseekVLTextRMSNorm,
)
from kerasformers.models.deepseek_vl.deepseek_vl_model import DeepseekVLVisionModel

from .deepseek_vl_hybrid_config import (
    DEEPSEEK_VL_HYBRID_CONFIG,
    DEEPSEEK_VL_HYBRID_WEIGHTS_URLS,
)
from .deepseek_vl_hybrid_layers import (
    DeepseekVLHybridAligner,
    DeepseekVLHybridSamEncoder,
    DeepseekVLSamVisionNeck,
    DeepseekVLSamVisionProj,
)

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekVLHybridModel(SubclassedBaseModel):
    """DeepSeek-VL Hybrid (7B) backbone: dual vision (SigLIP @384 + SAM @1024)
    + 3-way aligner + Llama-7B decoder.

    The low-res SigLIP tower and the high-res SAM/ViTDet tower each produce 576
    tokens; the SAM stream blends its final neck output with a necked
    intermediate global-attention state (scaled by a learned ``alpha``), and the
    :class:`DeepseekVLHybridAligner` concatenates the two streams into the text
    width and scatters them into the ``image_token_id`` slots of the decoder.
    Covers the HF ``model_type: "deepseek_vl_hybrid"`` checkpoints (the 7B
    chat/base repos). Returns raw features; use :class:`DeepseekVLHybridGenerate`
    for logits / text.
    """

    HF_MODEL_TYPE = "deepseek_vl_hybrid"
    BASE_MODEL_CONFIG = DEEPSEEK_VL_HYBRID_CONFIG
    BASE_WEIGHT_CONFIG = DEEPSEEK_VL_HYBRID_WEIGHTS_URLS

    def __init__(
        self,
        vocab_size=102400,
        embed_dim=4096,
        mlp_dim=11008,
        num_layers=30,
        num_heads=32,
        num_kv_heads=32,
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
        high_res_embed_dim=768,
        high_res_mlp_dim=3072,
        high_res_num_layers=12,
        high_res_num_heads=12,
        high_res_image_size=1024,
        high_res_patch_size=16,
        high_res_output_channels=256,
        high_res_window_size=14,
        high_res_global_attn_indexes=(2, 5, 8, 11),
        high_res_norm_eps=1e-6,
        image_token_id=100015,
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
        self.high_res_embed_dim = high_res_embed_dim
        self.high_res_mlp_dim = high_res_mlp_dim
        self.high_res_num_layers = high_res_num_layers
        self.high_res_num_heads = high_res_num_heads
        self.high_res_image_size = high_res_image_size
        self.high_res_patch_size = high_res_patch_size
        self.high_res_output_channels = high_res_output_channels
        self.high_res_window_size = high_res_window_size
        self.high_res_global_attn_indexes = tuple(high_res_global_attn_indexes)
        self.high_res_norm_eps = high_res_norm_eps
        self.image_token_id = image_token_id

        self.vision_model = DeepseekVLVisionModel(
            vision_embed_dim,
            vision_mlp_dim,
            vision_num_layers,
            vision_num_heads,
            image_size,
            patch_size,
            vision_norm_eps,
            name="vision_model",
        )
        self.high_res_vision_model = DeepseekVLHybridSamEncoder(
            high_res_embed_dim,
            high_res_num_layers,
            high_res_num_heads,
            high_res_mlp_dim,
            high_res_image_size,
            high_res_patch_size,
            high_res_output_channels,
            high_res_window_size,
            self.high_res_global_attn_indexes,
            high_res_norm_eps,
            name="high_res_vision_model",
        )
        self.high_res_vision_neck = DeepseekVLSamVisionNeck(
            high_res_output_channels, high_res_norm_eps, name="high_res_vision_neck"
        )
        self.high_res_vision_proj = DeepseekVLSamVisionProj(
            high_res_output_channels,
            image_size // patch_size,
            name="high_res_vision_proj",
        )
        self.high_res_vision_alpha = self.add_weight(
            shape=(1,),
            initializer="zeros",
            trainable=True,
            name="high_res_vision_alpha",
        )
        self.aligner = DeepseekVLHybridAligner(embed_dim, name="aligner")
        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            DeepseekVLTextDecoderLayer(
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
        self.final_norm = DeepseekVLTextRMSNorm(eps=norm_eps, name="final_norm")

    def get_high_res_features(self, high_res_pixel_values):
        last, global_state = self.high_res_vision_model(high_res_pixel_values)
        last = self.high_res_vision_proj(last)
        glob = self.high_res_vision_neck(global_state)
        glob = self.high_res_vision_proj(glob)
        out = last + glob * self.high_res_vision_alpha
        b = ops.shape(out)[0]
        return ops.reshape(out, (b, -1, ops.shape(out)[-1]))

    def get_image_features(self, pixel_values, high_res_pixel_values):
        low = self.vision_model(pixel_values)
        high = self.get_high_res_features(high_res_pixel_values)
        return self.aligner(low, high)

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

    def prepare_inputs(
        self, input_ids, pixel_values, high_res_pixel_values, attention_mask
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        inputs_embeds = self.token_embedding(input_ids)
        if pixel_values is not None:
            image_embeds = ops.reshape(
                self.get_image_features(pixel_values, high_res_pixel_values),
                (-1, self.embed_dim),
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
            input_ids,
            inputs.get("pixel_values"),
            inputs.get("high_res_pixel_values"),
            inputs.get("attention_mask"),
        )
        cos, sin = self.rope_tables(position_ids)
        attn_mask = self.causal_mask(seq, inputs.get("attention_mask"))
        for layer in self.decoder_layers:
            hidden = layer(hidden, cos, sin, attention_mask=attn_mask)
        return self.final_norm(hidden)

    def call(self, inputs):
        return {"last_hidden_state": self.forward_features(inputs)}

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        # Subclassed model: build the graph with a dummy dual-resolution forward
        # (image-placeholder tokens + 384 low-res + 1024 high-res zeros) before
        # loading the released sharded .weights.json (7B float32 > 2 GB).
        entry = cls.BASE_WEIGHT_CONFIG.get(variant, {})
        url = entry.get("url") if isinstance(entry, dict) else entry
        if not (load_weights and url):
            return super().from_release(
                variant,
                load_weights=load_weights,
                skip_mismatch=skip_mismatch,
                **kwargs,
            )
        model = super().from_release(variant, load_weights=False, **kwargs)
        num_patches = (model.image_size // model.patch_size) ** 2
        model(
            {
                "input_ids": ops.full(
                    (1, num_patches), model.image_token_id, dtype="int32"
                ),
                "pixel_values": ops.zeros(
                    (1, model.image_size, model.image_size, 3), dtype="float32"
                ),
                "high_res_pixel_values": ops.zeros(
                    (1, model.high_res_image_size, model.high_res_image_size, 3),
                    dtype="float32",
                ),
            }
        )
        cls.load_weights_from_url(model, url, skip_mismatch)
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config["text_config"]
        vision = hf_config["vision_config"]
        high = hf_config["high_res_vision_config"]
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
            "high_res_embed_dim": high["hidden_size"],
            "high_res_mlp_dim": high.get("mlp_dim", high.get("intermediate_size")),
            "high_res_num_layers": high["num_hidden_layers"],
            "high_res_num_heads": high["num_attention_heads"],
            "high_res_image_size": high.get("image_size", 1024),
            "high_res_patch_size": high.get("patch_size", 16),
            "high_res_output_channels": high.get("output_channels", 256),
            "high_res_window_size": high.get("window_size", 14),
            "high_res_global_attn_indexes": tuple(
                high.get("global_attn_indexes", (2, 5, 8, 11))
            ),
            "high_res_norm_eps": high.get("layer_norm_eps", 1e-6),
            "image_token_id": hf_config.get("image_token_id", 100015),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_deepseek_vl_hybrid_hf_to_keras import (
            transfer_deepseek_vl_hybrid_weights,
        )

        transfer_deepseek_vl_hybrid_weights(keras_model, hf_state_dict)

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
                "high_res_embed_dim": self.high_res_embed_dim,
                "high_res_mlp_dim": self.high_res_mlp_dim,
                "high_res_num_layers": self.high_res_num_layers,
                "high_res_num_heads": self.high_res_num_heads,
                "high_res_image_size": self.high_res_image_size,
                "high_res_patch_size": self.high_res_patch_size,
                "high_res_output_channels": self.high_res_output_channels,
                "high_res_window_size": self.high_res_window_size,
                "high_res_global_attn_indexes": self.high_res_global_attn_indexes,
                "high_res_norm_eps": self.high_res_norm_eps,
                "image_token_id": self.image_token_id,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekVLHybridGenerate(DeepseekVLHybridModel, BaseGeneration):
    """DeepSeek-VL Hybrid with an LM head + fast ``.generate()``.

    Adds a bias-free ``lm_head`` on top of :class:`DeepseekVLHybridModel`.
    ``build_cache`` runs both vision towers + aligner + fused prefill ONCE
    (consuming ``pixel_values`` and ``high_res_pixel_values``), then
    ``call_with_cache`` does text-only decode:

        gen.generate(input_ids, pixel_values=..., high_res_pixel_values=...)
    """

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

    def build_cache(
        self,
        token_ids,
        padding_mask,
        max_len,
        pixel_values=None,
        high_res_pixel_values=None,
    ):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        hidden, position_ids = self.prepare_inputs(
            token_ids, pixel_values, high_res_pixel_values, padding_mask
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
