import keras
from keras import layers, ops

from kmodels.model_registry import register_model
from kmodels.weight_utils import load_weights_from_config

from .config import OWLVIT_MODEL_CONFIG, OWLVIT_WEIGHTS_CONFIG
from .owlvit_layers import (
    OwlViTBoxPredictionHead,
    OwlViTClassPredictionHead,
    OwlViTTextTransformer,
    OwlViTVisionTransformer,
    compute_box_bias,
)


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTCore(keras.Model):
    """OWL-ViT (Open-vocabulary Object Detection with Vision Transformers).

    OWL-ViT performs open-vocabulary object detection by composing a
    CLIP-style vision transformer with a CLIP-style causal text
    transformer. Each patch of the image emits one box and a
    per-text-query similarity score, so the set of detection classes
    is the set of text queries provided at inference time rather than
    a fixed softmax head.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        vision_image_size: Integer, square image input edge in pixels.
        vision_patch_size: Integer, ViT patch edge in pixels.
        vision_hidden_size: Integer, hidden size of the vision tower.
        vision_intermediate_size: Integer, MLP intermediate size of the
            vision tower.
        vision_num_hidden_layers: Integer, number of vision transformer
            layers.
        vision_num_attention_heads: Integer, number of vision attention
            heads.
        text_max_position_embeddings: Integer, maximum text sequence
            length.
        text_hidden_size: Integer, hidden size of the text tower.
        text_intermediate_size: Integer, MLP intermediate size of the
            text tower.
        text_num_hidden_layers: Integer, number of text transformer
            layers.
        text_num_attention_heads: Integer, number of text attention
            heads.
        text_vocab_size: Integer, text vocabulary size.
        projection_dim: Integer, joint contrastive projection size for
            ``visual_projection`` / ``text_projection``.
        logit_scale_init_value: Float, initial value for the joint
            contrastive logit scale.
        layer_norm_eps: Float, layer normalization epsilon.
        hidden_act: String, activation used in transformer MLPs.
            ``"quick_gelu"`` matches HF defaults.
        name: String, model name.

    Inputs:
        Dictionary with keys:
        - ``"pixel_values"``: ``(B, H, W, 3)`` channels-last image batch.
        - ``"input_ids"``: ``(B*Q, T)`` text token ids for ``Q`` queries
          per image.

    Outputs:
        Dictionary with keys:
        - ``"logits"``: ``(B, num_patches, Q)`` per-query similarity.
        - ``"pred_boxes"``: ``(B, num_patches, 4)`` normalized
          ``(cx, cy, w, h)``.
        - ``"text_embeds"``: ``(B, Q, projection_dim)`` L2-normalized
          text embeddings.
        - ``"image_embeds"``: ``(B, h_patches, w_patches, vision_hidden)``
          per-patch image features.
        - ``"class_embeds"``: ``(B, num_patches, text_hidden)`` per-patch
          features projected into the text space.
    """

    def __init__(
        self,
        vision_image_size: int,
        vision_patch_size: int,
        vision_hidden_size: int,
        vision_intermediate_size: int,
        vision_num_hidden_layers: int,
        vision_num_attention_heads: int,
        text_max_position_embeddings: int,
        text_hidden_size: int,
        text_intermediate_size: int,
        text_num_hidden_layers: int,
        text_num_attention_heads: int,
        text_vocab_size: int,
        projection_dim: int,
        logit_scale_init_value: float = 2.6592,
        layer_norm_eps: float = 1e-5,
        hidden_act: str = "quick_gelu",
        name: str = "OwlViTCore",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.vision_image_size = vision_image_size
        self.vision_patch_size = vision_patch_size
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.text_max_position_embeddings = text_max_position_embeddings
        self.text_hidden_size = text_hidden_size
        self.text_intermediate_size = text_intermediate_size
        self.text_num_hidden_layers = text_num_hidden_layers
        self.text_num_attention_heads = text_num_attention_heads
        self.text_vocab_size = text_vocab_size
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

        self.num_patches_h = vision_image_size // vision_patch_size
        self.num_patches_w = vision_image_size // vision_patch_size

        self.vision_model = OwlViTVisionTransformer(
            hidden_size=vision_hidden_size,
            image_size=vision_image_size,
            patch_size=vision_patch_size,
            num_hidden_layers=vision_num_hidden_layers,
            num_heads=vision_num_attention_heads,
            intermediate_size=vision_intermediate_size,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
            name="vision_model",
        )
        self.text_model = OwlViTTextTransformer(
            vocab_size=text_vocab_size,
            hidden_size=text_hidden_size,
            max_position_embeddings=text_max_position_embeddings,
            num_hidden_layers=text_num_hidden_layers,
            num_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
            name="text_model",
        )
        self.visual_projection = layers.Dense(
            projection_dim, use_bias=False, name="visual_projection"
        )
        self.text_projection = layers.Dense(
            projection_dim, use_bias=False, name="text_projection"
        )

        self.class_head = OwlViTClassPredictionHead(
            query_dim=vision_hidden_size,
            out_dim=text_hidden_size,
            name="class_head",
        )
        self.box_head = OwlViTBoxPredictionHead(
            hidden_size=vision_hidden_size,
            out_dim=4,
            name="box_head",
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps,
            name="layer_norm",
        )

        self._box_bias = ops.convert_to_tensor(
            compute_box_bias(self.num_patches_h, self.num_patches_w)
        )

    def build(self, input_shape):
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(),
            initializer=keras.initializers.Constant(self.logit_scale_init_value),
            trainable=True,
        )
        super().build(input_shape)

    def get_text_features(self, input_ids, attention_mask=None):
        del attention_mask
        _, pooled = self.text_model(input_ids)
        return self.text_projection(pooled)

    def get_image_features(self, pixel_values):
        last_hidden = self.vision_model(pixel_values)
        pooled = self.vision_model.post_layernorm(last_hidden[:, 0, :])
        return self.visual_projection(pooled)

    def image_text_embedder(self, pixel_values, input_ids):
        vision_last_hidden = self.vision_model(pixel_values)
        image_embeds = self.vision_model.post_layernorm(vision_last_hidden)

        cls = image_embeds[:, :1, :]
        cls_broadcast = ops.broadcast_to(cls, ops.shape(image_embeds[:, :-1, :]))
        patch_embeds = image_embeds[:, 1:, :] * cls_broadcast
        patch_embeds = self.layer_norm(patch_embeds)

        b = ops.shape(patch_embeds)[0]
        feature_map = ops.reshape(
            patch_embeds,
            (b, self.num_patches_h, self.num_patches_w, self.vision_hidden_size),
        )

        _, text_pooled = self.text_model(input_ids)
        query_embeds = self.text_projection(text_pooled)

        return query_embeds, feature_map

    def call(self, inputs, training=None):
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]

        query_embeds, feature_map = self.image_text_embedder(pixel_values, input_ids)

        b = ops.shape(feature_map)[0]
        num_patches = self.num_patches_h * self.num_patches_w
        image_feats = ops.reshape(
            feature_map, (b, num_patches, self.vision_hidden_size)
        )

        norm = ops.sqrt(
            ops.sum(query_embeds * query_embeds, axis=-1, keepdims=True) + 1e-12
        )
        query_embeds = query_embeds / norm

        max_text_queries = ops.shape(input_ids)[0] // b
        query_embeds = ops.reshape(
            query_embeds, (b, max_text_queries, self.text_hidden_size)
        )
        input_ids_b = ops.reshape(
            input_ids, (b, max_text_queries, ops.shape(input_ids)[-1])
        )
        query_mask = input_ids_b[..., 0] > 0

        pred_logits, class_embeds = self.class_head(
            image_feats, query_embeds, query_mask
        )

        pred_boxes = self.box_head(image_feats)
        pred_boxes = pred_boxes + ops.cast(self._box_bias, pred_boxes.dtype)
        pred_boxes = ops.sigmoid(pred_boxes)

        return {
            "logits": pred_logits,
            "pred_boxes": pred_boxes,
            "text_embeds": query_embeds,
            "image_embeds": feature_map,
            "class_embeds": class_embeds,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_image_size": self.vision_image_size,
                "vision_patch_size": self.vision_patch_size,
                "vision_hidden_size": self.vision_hidden_size,
                "vision_intermediate_size": self.vision_intermediate_size,
                "vision_num_hidden_layers": self.vision_num_hidden_layers,
                "vision_num_attention_heads": self.vision_num_attention_heads,
                "text_max_position_embeddings": self.text_max_position_embeddings,
                "text_hidden_size": self.text_hidden_size,
                "text_intermediate_size": self.text_intermediate_size,
                "text_num_hidden_layers": self.text_num_hidden_layers,
                "text_num_attention_heads": self.text_num_attention_heads,
                "text_vocab_size": self.text_vocab_size,
                "projection_dim": self.projection_dim,
                "logit_scale_init_value": self.logit_scale_init_value,
                "layer_norm_eps": self.layer_norm_eps,
                "hidden_act": self.hidden_act,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_owlvit_model(variant, weights="owlvit", name=None, **kwargs):
    cfg = OWLVIT_MODEL_CONFIG[variant]
    model = OwlViTCore(
        vision_image_size=cfg["vision_image_size"],
        vision_patch_size=cfg["vision_patch_size"],
        vision_hidden_size=cfg["vision_hidden_size"],
        vision_intermediate_size=cfg["vision_intermediate_size"],
        vision_num_hidden_layers=cfg["vision_num_hidden_layers"],
        vision_num_attention_heads=cfg["vision_num_attention_heads"],
        text_max_position_embeddings=cfg["text_max_position_embeddings"],
        text_hidden_size=cfg["text_hidden_size"],
        text_intermediate_size=cfg["text_intermediate_size"],
        text_num_hidden_layers=cfg["text_num_hidden_layers"],
        text_num_attention_heads=cfg["text_num_attention_heads"],
        text_vocab_size=cfg["text_vocab_size"],
        projection_dim=cfg["projection_dim"],
        logit_scale_init_value=cfg["logit_scale_init_value"],
        layer_norm_eps=cfg["layer_norm_eps"],
        hidden_act=cfg["hidden_act"],
        name=name or variant,
        **kwargs,
    )

    image_size = cfg["vision_image_size"]
    text_len = cfg["text_max_position_embeddings"]
    dummy_pixel = ops.zeros((1, image_size, image_size, 3), dtype="float32")
    dummy_ids = ops.ones((1, text_len), dtype="int32")
    _ = model({"pixel_values": dummy_pixel, "input_ids": dummy_ids})
    _ = model.get_image_features(dummy_pixel)

    if weights in OWLVIT_WEIGHTS_CONFIG.get(variant, {}):
        url = OWLVIT_WEIGHTS_CONFIG[variant][weights].get("url", "")
        if url:
            load_weights_from_config(variant, weights, model, OWLVIT_WEIGHTS_CONFIG)
        else:
            print(
                f"Weight URL for '{weights}' is not yet available. "
                "Use the conversion script to generate weights."
            )
    elif weights is not None and weights != "owlvit":
        model.load_weights(weights)
    else:
        if weights == "owlvit":
            print(
                "OWL-ViT weights URL not yet configured. "
                "Run convert_owlvit_hf_to_keras.py to generate weights, "
                "then pass the .weights.h5 file path."
            )

    return model


@register_model
def OwlViTBasePatch32(weights="owlvit", name="OwlViTBasePatch32", **kwargs):
    return _create_owlvit_model(
        "OwlViTBasePatch32", weights=weights, name=name, **kwargs
    )


@register_model
def OwlViTBasePatch16(weights="owlvit", name="OwlViTBasePatch16", **kwargs):
    return _create_owlvit_model(
        "OwlViTBasePatch16", weights=weights, name=name, **kwargs
    )


@register_model
def OwlViTLargePatch14(weights="owlvit", name="OwlViTLargePatch14", **kwargs):
    return _create_owlvit_model(
        "OwlViTLargePatch14", weights=weights, name=name, **kwargs
    )
