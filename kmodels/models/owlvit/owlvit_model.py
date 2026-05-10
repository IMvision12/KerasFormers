import keras
from keras import layers, ops

from kmodels.model_registry import register_model
from kmodels.weight_utils import load_weights_from_config

from .config import OWLVIT_MODEL_CONFIG, OWLVIT_WEIGHTS_CONFIG
from .owlvit_layers import (
    OwlViTAttention,
    OwlViTSplitBatchQueries,
    OwlViTTextEmbeddings,
    OwlViTVisionEmbeddings,
    compute_box_bias,
    quick_gelu,
)


def owlvit_mlp(
    x,
    hidden_size,
    intermediate_size,
    hidden_act,
    block_prefix,
):
    """Two-layer MLP block (``fc1`` → activation → ``fc2``).

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        x: Input tensor of shape
            ``(batch_size, seq_len, hidden_size)``.
        hidden_size: Integer, model dimension.
        intermediate_size: Integer, MLP expansion dimension.
        hidden_act: String, activation function. ``"quick_gelu"`` or
            ``"gelu"``.
        block_prefix: String, name prefix for the dense layers.

    Returns:
        Tensor of shape ``(batch_size, seq_len, hidden_size)``.
    """
    x = layers.Dense(intermediate_size, name=f"{block_prefix}_fc1")(x)
    if hidden_act == "quick_gelu":
        x = quick_gelu(x)
    elif hidden_act == "gelu":
        x = ops.gelu(x, approximate=False)
    else:
        x = keras.activations.get(hidden_act)(x)
    x = layers.Dense(hidden_size, name=f"{block_prefix}_fc2")(x)
    return x


def owlvit_encoder_layer(
    x,
    attention_mask,
    hidden_size,
    num_heads,
    intermediate_size,
    layer_norm_eps,
    hidden_act,
    block_prefix,
):
    """Pre-norm transformer block: LN → SA → residual → LN → MLP → residual.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        x: Input tensor of shape
            ``(batch_size, seq_len, hidden_size)``.
        attention_mask: Optional additive attention mask broadcastable to
            ``(batch_size, num_heads, seq_len, seq_len)``.
        hidden_size: Integer, model dimension.
        num_heads: Integer, number of attention heads.
        intermediate_size: Integer, MLP expansion dimension.
        layer_norm_eps: Float, layer normalization epsilon.
        hidden_act: String, MLP activation.
        block_prefix: String, name prefix for all sub-layers.

    Returns:
        Tensor of shape ``(batch_size, seq_len, hidden_size)``.
    """
    residual = x
    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_layer_norm1"
    )(x)
    x = OwlViTAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        name=f"{block_prefix}_self_attn",
    )(x, attention_mask=attention_mask)
    x = layers.Add(name=f"{block_prefix}_sa_residual")([residual, x])

    residual = x
    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_layer_norm2"
    )(x)
    x = owlvit_mlp(
        x,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        block_prefix=f"{block_prefix}_mlp",
    )
    return layers.Add(name=f"{block_prefix}_ff_residual")([residual, x])


def owlvit_encoder(
    x,
    attention_mask,
    num_layers,
    hidden_size,
    num_heads,
    intermediate_size,
    layer_norm_eps,
    hidden_act,
    block_prefix,
):
    """Stack of ``num_layers`` ``owlvit_encoder_layer`` blocks.

    Args:
        x: Input tensor.
        attention_mask: Optional additive attention mask.
        num_layers: Integer, number of stacked encoder layers.
        hidden_size: Integer, model dimension.
        num_heads: Integer, attention heads per layer.
        intermediate_size: Integer, MLP expansion dimension per layer.
        layer_norm_eps: Float, layer normalization epsilon.
        hidden_act: String, MLP activation.
        block_prefix: String, name prefix; each layer is named
            ``f"{block_prefix}_layers_{i}_..."``.

    Returns:
        Output tensor with the same shape as ``x``.
    """
    for i in range(num_layers):
        x = owlvit_encoder_layer(
            x,
            attention_mask=attention_mask,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
            block_prefix=f"{block_prefix}_layers_{i}",
        )
    return x


def build_owlvit_vision_transformer(
    pixel_values,
    hidden_size,
    image_size,
    patch_size,
    num_hidden_layers,
    num_heads,
    intermediate_size,
    layer_norm_eps,
    hidden_act,
    block_prefix,
):
    """OWL-ViT vision tower: embeddings → pre LN → encoder → post LN.

    Returns:
        Tensor of shape ``(batch_size, num_patches + 1, hidden_size)``.
    """
    x = OwlViTVisionEmbeddings(
        hidden_size=hidden_size,
        image_size=image_size,
        patch_size=patch_size,
        name=f"{block_prefix}_embeddings",
    )(pixel_values)
    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_pre_layernorm"
    )(x)
    x = owlvit_encoder(
        x,
        attention_mask=None,
        num_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        layer_norm_eps=layer_norm_eps,
        hidden_act=hidden_act,
        block_prefix=block_prefix,
    )
    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_post_layernorm"
    )(x)
    return x


def build_owlvit_text_transformer(
    input_ids,
    vocab_size,
    hidden_size,
    max_position_embeddings,
    num_hidden_layers,
    num_heads,
    intermediate_size,
    layer_norm_eps,
    hidden_act,
    block_prefix,
):
    """OWL-ViT text tower: embeddings → causal encoder → final LN → pool.

    Pools by gathering the per-row argmax of ``input_ids`` (matches
    HF's ``"EOT is the highest token id"`` convention).

    Returns:
        Pooled tensor of shape ``(batch_size, hidden_size)``.
    """
    x = OwlViTTextEmbeddings(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=max_position_embeddings,
        name=f"{block_prefix}_embeddings",
    )(input_ids)

    seq_len = max_position_embeddings
    i = ops.arange(seq_len)[:, None]
    j = ops.arange(seq_len)[None, :]
    causal = ops.where(j > i, ops.cast(-1e9, "float32"), ops.cast(0.0, "float32"))
    causal = ops.reshape(causal, (1, 1, seq_len, seq_len))

    x = owlvit_encoder(
        x,
        attention_mask=causal,
        num_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        layer_norm_eps=layer_norm_eps,
        hidden_act=hidden_act,
        block_prefix=block_prefix,
    )
    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{block_prefix}_final_layer_norm"
    )(x)

    pool_indices = ops.cast(ops.argmax(input_ids, axis=-1), "int32")
    gather = ops.expand_dims(ops.expand_dims(pool_indices, -1), -1)
    pooled = ops.take_along_axis(x, gather, axis=1)
    pooled = ops.squeeze(pooled, axis=1)
    return pooled


def owlvit_box_predictor(image_features, hidden_size, block_prefix):
    """3-layer MLP predicting raw ``(cx, cy, w, h)`` per patch.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        image_features: Tensor of shape
            ``(batch_size, num_patches, hidden_size)``.
        hidden_size: Integer, hidden size of the MLP.
        block_prefix: String, name prefix for the dense layers.

    Returns:
        Tensor of shape ``(batch_size, num_patches, 4)``.
    """
    x = layers.Dense(hidden_size, name=f"{block_prefix}_dense0")(image_features)
    x = ops.gelu(x, approximate=False)
    x = layers.Dense(hidden_size, name=f"{block_prefix}_dense1")(x)
    x = ops.gelu(x, approximate=False)
    x = layers.Dense(4, name=f"{block_prefix}_dense2")(x)
    return x


def owlvit_class_predictor(
    image_embeds,
    query_embeds,
    query_mask,
    out_dim,
    block_prefix,
):
    """Text-conditional class predictor.

    Projects per-patch image features into the text dimension, takes
    the L2-normalized cosine similarity against each text query, and
    applies a per-patch learned shift+scale.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        image_embeds: ``(batch_size, num_patches, vision_hidden)``.
        query_embeds: ``(batch_size, num_queries, out_dim)``.
        query_mask: Bool tensor ``(batch_size, num_queries)`` masking
            padded queries to ``-inf``.
        out_dim: Integer, hidden size of the text features the
            similarity is computed against.
        block_prefix: String, name prefix for the dense layers.

    Returns:
        Tuple ``(pred_logits, image_class_embeds)`` where
        ``pred_logits`` has shape
        ``(batch_size, num_patches, num_queries)``.
    """
    image_class_embeds = layers.Dense(out_dim, name=f"{block_prefix}_dense0")(
        image_embeds
    )
    image_norm = (
        ops.sqrt(
            ops.sum(image_class_embeds * image_class_embeds, axis=-1, keepdims=True)
            + 1e-12
        )
        + 1e-6
    )
    image_class_embeds_n = image_class_embeds / image_norm

    query_norm = (
        ops.sqrt(ops.sum(query_embeds * query_embeds, axis=-1, keepdims=True) + 1e-12)
        + 1e-6
    )
    query_embeds_n = query_embeds / query_norm

    pred_logits = ops.matmul(
        image_class_embeds_n, ops.transpose(query_embeds_n, (0, 2, 1))
    )

    logit_shift = layers.Dense(1, name=f"{block_prefix}_logit_shift")(image_embeds)
    logit_scale_pred = layers.Dense(1, name=f"{block_prefix}_logit_scale")(image_embeds)
    logit_scale_pred = ops.elu(logit_scale_pred) + 1.0

    pred_logits = (pred_logits + logit_shift) * logit_scale_pred

    if query_mask is not None:
        mask = ops.expand_dims(ops.cast(query_mask, "bool"), axis=-2)
        very_neg = ops.cast(ops.full_like(pred_logits, -1e30), pred_logits.dtype)
        pred_logits = ops.where(mask, pred_logits, very_neg)

    return pred_logits, image_class_embeds


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

        image_embeds = build_owlvit_vision_transformer(
            pixel_values,
            hidden_size=vision_hidden_size,
            image_size=vision_image_size,
            patch_size=vision_patch_size,
            num_hidden_layers=vision_num_hidden_layers,
            num_heads=vision_num_attention_heads,
            intermediate_size=vision_intermediate_size,
            layer_norm_eps=self.LAYER_NORM_EPS,
            hidden_act=self.HIDDEN_ACT,
            block_prefix="vision_model",
        )

        cls = image_embeds[:, :1, :]
        patch_embeds = image_embeds[:, 1:, :] * cls
        patch_embeds = layers.LayerNormalization(
            epsilon=self.LAYER_NORM_EPS, name="layer_norm"
        )(patch_embeds)

        feature_map = ops.reshape(
            patch_embeds,
            (-1, num_patches_h, num_patches_w, vision_hidden_size),
        )
        image_feats = ops.reshape(patch_embeds, (-1, num_patches, vision_hidden_size))

        text_pooled = build_owlvit_text_transformer(
            input_ids,
            vocab_size=self.TEXT_VOCAB_SIZE,
            hidden_size=text_hidden_size,
            max_position_embeddings=self.TEXT_MAX_POSITION_EMBEDDINGS,
            num_hidden_layers=self.TEXT_NUM_HIDDEN_LAYERS,
            num_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            layer_norm_eps=self.LAYER_NORM_EPS,
            hidden_act=self.HIDDEN_ACT,
            block_prefix="text_model",
        )
        query_embeds = layers.Dense(
            projection_dim, use_bias=False, name="text_projection"
        )(text_pooled)

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

        pred_logits, class_embeds = owlvit_class_predictor(
            image_feats,
            query_embeds,
            query_mask,
            out_dim=text_hidden_size,
            block_prefix="class_head",
        )

        box_bias = ops.cast(compute_box_bias(num_patches_h, num_patches_w), "float32")
        pred_boxes = owlvit_box_predictor(
            image_feats,
            hidden_size=vision_hidden_size,
            block_prefix="box_head",
        )
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
