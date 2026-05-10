import keras
from keras import layers, ops

from kmodels.model_registry import register_model
from kmodels.weight_utils import load_weights_from_config

from .config import OWLVIT_MODEL_CONFIG, OWLVIT_WEIGHTS_CONFIG
from .owlvit_layers import (
    OwlViTBoxPredictionHead,
    OwlViTClassPredictionHead,
    OwlViTSplitBatchQueries,
    OwlViTTextTransformer,
    OwlViTVisionTransformer,
    compute_box_bias,
)


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViT(keras.Model):
    """OWL-ViT (Open-vocabulary Object Detection with Vision Transformers).

    OWL-ViT performs open-vocabulary object detection by composing a
    CLIP-style vision transformer with a CLIP-style causal text
    transformer. Each patch of the image emits one box and a
    per-text-query similarity score, so the set of detection classes
    is the set of text queries provided at inference time rather than
    a fixed softmax head.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Hyperparameters that are constant across all published OWL-ViT
    variants are exposed as class-level constants
    (``TEXT_MAX_POSITION_EMBEDDINGS``, ``TEXT_NUM_HIDDEN_LAYERS``,
    ``TEXT_VOCAB_SIZE``, ``LAYER_NORM_EPS``, ``HIDDEN_ACT``) rather
    than ``__init__`` args.

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
        text_hidden_size: Integer, hidden size of the text tower.
        text_intermediate_size: Integer, MLP intermediate size of the
            text tower.
        text_num_attention_heads: Integer, number of text attention
            heads.
        projection_dim: Integer, joint contrastive projection size for
            ``text_projection``.
        weights: ``None``, a config key (e.g. ``"owlvit"``), or a path
            to a ``.weights.h5`` file.
        input_shape: Optional ``(H, W, 3)`` for the image input.
        text_input_shape: Optional ``(T,)`` for the input-ids input.
        name: String, model name.

    Returns:
        A ``keras.Model`` with dict outputs:
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

    TEXT_MAX_POSITION_EMBEDDINGS = 16
    TEXT_NUM_HIDDEN_LAYERS = 12
    TEXT_VOCAB_SIZE = 49408
    LAYER_NORM_EPS = 1e-5
    HIDDEN_ACT = "quick_gelu"

    def __init__(
        self,
        vision_image_size,
        vision_patch_size,
        vision_hidden_size,
        vision_intermediate_size,
        vision_num_hidden_layers,
        vision_num_attention_heads,
        text_hidden_size,
        text_intermediate_size,
        text_num_attention_heads,
        projection_dim,
        weights=None,
        input_shape=None,
        text_input_shape=None,
        name="OwlViT",
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (vision_image_size, vision_image_size, 3)
        if text_input_shape is None:
            text_input_shape = (self.TEXT_MAX_POSITION_EMBEDDINGS,)

        num_patches_h = vision_image_size // vision_patch_size
        num_patches_w = vision_image_size // vision_patch_size
        num_patches = num_patches_h * num_patches_w

        pixel_values = layers.Input(shape=input_shape, name="pixel_values")
        input_ids = layers.Input(
            shape=text_input_shape, dtype="int32", name="input_ids"
        )

        vision_model = OwlViTVisionTransformer(
            hidden_size=vision_hidden_size,
            image_size=vision_image_size,
            patch_size=vision_patch_size,
            num_hidden_layers=vision_num_hidden_layers,
            num_heads=vision_num_attention_heads,
            intermediate_size=vision_intermediate_size,
            layer_norm_eps=self.LAYER_NORM_EPS,
            hidden_act=self.HIDDEN_ACT,
            name="vision_model",
        )
        text_model = OwlViTTextTransformer(
            vocab_size=self.TEXT_VOCAB_SIZE,
            hidden_size=text_hidden_size,
            max_position_embeddings=self.TEXT_MAX_POSITION_EMBEDDINGS,
            num_hidden_layers=self.TEXT_NUM_HIDDEN_LAYERS,
            num_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            layer_norm_eps=self.LAYER_NORM_EPS,
            hidden_act=self.HIDDEN_ACT,
            name="text_model",
        )
        text_projection = layers.Dense(
            projection_dim, use_bias=False, name="text_projection"
        )
        layer_norm = layers.LayerNormalization(
            epsilon=self.LAYER_NORM_EPS, name="layer_norm"
        )
        class_head = OwlViTClassPredictionHead(
            query_dim=vision_hidden_size,
            out_dim=text_hidden_size,
            name="class_head",
        )
        box_head = OwlViTBoxPredictionHead(
            hidden_size=vision_hidden_size,
            out_dim=4,
            name="box_head",
        )

        vision_last_hidden = vision_model(pixel_values)
        image_embeds = vision_model.post_layernorm(vision_last_hidden)

        cls = image_embeds[:, :1, :]
        patch_embeds = image_embeds[:, 1:, :] * cls
        patch_embeds = layer_norm(patch_embeds)

        feature_map = ops.reshape(
            patch_embeds,
            (-1, num_patches_h, num_patches_w, vision_hidden_size),
        )
        image_feats = ops.reshape(patch_embeds, (-1, num_patches, vision_hidden_size))

        _, text_pooled = text_model(input_ids)
        query_embeds = text_projection(text_pooled)

        norm = ops.sqrt(
            ops.sum(query_embeds * query_embeds, axis=-1, keepdims=True) + 1e-12
        )
        query_embeds = query_embeds / norm

        query_embeds = OwlViTSplitBatchQueries(name="split_text_embeds")(
            query_embeds, patch_embeds
        )
        input_ids_b = OwlViTSplitBatchQueries(name="split_input_ids")(
            input_ids, patch_embeds
        )
        query_mask = input_ids_b[..., 0] > 0

        pred_logits, class_embeds = class_head(image_feats, query_embeds, query_mask)

        box_bias = ops.cast(compute_box_bias(num_patches_h, num_patches_w), "float32")
        pred_boxes = box_head(image_feats)
        pred_boxes = pred_boxes + ops.cast(box_bias, pred_boxes.dtype)
        pred_boxes = ops.sigmoid(pred_boxes)

        outputs = {
            "logits": pred_logits,
            "pred_boxes": pred_boxes,
            "text_embeds": query_embeds,
            "image_embeds": feature_map,
            "class_embeds": class_embeds,
        }
        inputs = {"pixel_values": pixel_values, "input_ids": input_ids}

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.vision_image_size = vision_image_size
        self.vision_patch_size = vision_patch_size
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.text_hidden_size = text_hidden_size
        self.text_intermediate_size = text_intermediate_size
        self.text_num_attention_heads = text_num_attention_heads
        self.projection_dim = projection_dim
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self._input_shape_arg = input_shape
        self._text_input_shape_arg = text_input_shape

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
                "text_hidden_size": self.text_hidden_size,
                "text_intermediate_size": self.text_intermediate_size,
                "text_num_attention_heads": self.text_num_attention_heads,
                "projection_dim": self.projection_dim,
                "input_shape": self._input_shape_arg,
                "text_input_shape": self._text_input_shape_arg,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_owlvit_model(variant, weights="owlvit", name=None, **kwargs):
    cfg = OWLVIT_MODEL_CONFIG[variant]

    model = OwlViT(
        vision_image_size=cfg["vision_image_size"],
        vision_patch_size=cfg["vision_patch_size"],
        vision_hidden_size=cfg["vision_hidden_size"],
        vision_intermediate_size=cfg["vision_intermediate_size"],
        vision_num_hidden_layers=cfg["vision_num_hidden_layers"],
        vision_num_attention_heads=cfg["vision_num_attention_heads"],
        text_hidden_size=cfg["text_hidden_size"],
        text_intermediate_size=cfg["text_intermediate_size"],
        text_num_attention_heads=cfg["text_num_attention_heads"],
        projection_dim=cfg["projection_dim"],
        weights=weights,
        name=name or variant,
        **kwargs,
    )

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
