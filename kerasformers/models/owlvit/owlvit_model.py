import keras
from keras import layers, ops

from kerasformers.base import BaseModel

from .config import OWLVIT_CONFIG, OWLVIT_WEIGHTS
from .owlvit_layers import (
    OwlViTAttention,
    OwlViTSplitBatchQueries,
    OwlViTTextEmbeddings,
    OwlViTVisionEmbeddings,
    compute_box_bias,
    quick_gelu,
)


def owlvit_mlp(x, hidden_size, intermediate_size, block_prefix):
    """Two-layer feed-forward MLP block (``fc1`` → quick_gelu → ``fc2``).

    The CLIP-style MLP used in both the vision and text towers: a Dense
    expanding to ``intermediate_size``, ``quick_gelu`` activation, and
    a Dense projecting back to ``hidden_size``.

    Reference:
        - `Simple Open-Vocabulary Object Detection with Vision Transformers
          <https://arxiv.org/abs/2205.06230>`_

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_size)``.
        hidden_size: Tower hidden dimension (also the output dim).
        intermediate_size: FFN intermediate dimension.
        block_prefix: Layer name prefix.

    Returns:
        Tensor of shape ``(B, seq_len, hidden_size)``.
    """
    x = layers.Dense(intermediate_size, name=f"{block_prefix}_fc1")(x)
    x = quick_gelu(x)
    x = layers.Dense(hidden_size, name=f"{block_prefix}_fc2")(x)
    return x


def owlvit_transformer_block(
    x,
    attention_mask,
    num_layers,
    hidden_size,
    num_heads,
    intermediate_size,
    block_prefix,
):
    """Stack of pre-norm transformer blocks shared by the vision and text towers.

    Each layer applies pre-norm self-attention (LayerNorm →
    :class:`OwlViTAttention` → residual) followed by pre-norm MLP
    (LayerNorm → :func:`owlvit_mlp` → residual). The vision tower
    passes ``attention_mask=None`` for bidirectional attention; the
    text tower passes a causal mask for autoregressive-style attention.

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_size)``.
        attention_mask: Optional additive attention mask broadcastable
            to ``(B, num_heads, seq_len, seq_len)``. Use ``None`` for
            full attention.
        num_layers: Number of stacked transformer layers.
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        block_prefix: Layer name prefix shared across all layers in
            the stack.

    Returns:
        Tensor of shape ``(B, seq_len, hidden_size)``.
    """
    for i in range(num_layers):
        prefix = f"{block_prefix}_layers_{i}"
        residual = x
        x = layers.LayerNormalization(epsilon=1e-5, name=f"{prefix}_layer_norm1")(x)
        x = OwlViTAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            name=f"{prefix}_self_attn",
        )(x, attention_mask=attention_mask)
        x = layers.Add(name=f"{prefix}_sa_residual")([residual, x])

        residual = x
        x = layers.LayerNormalization(epsilon=1e-5, name=f"{prefix}_layer_norm2")(x)
        x = owlvit_mlp(
            x,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            block_prefix=f"{prefix}_mlp",
        )
        x = layers.Add(name=f"{prefix}_ff_residual")([residual, x])
    return x


def owlvit_vision_transformer(
    pixel_values,
    hidden_size,
    image_size,
    patch_size,
    num_hidden_layers,
    num_heads,
    intermediate_size,
    block_prefix,
):
    """OWL-ViT vision tower: patch embeddings → pre-LN → encoder → post-LN.

    Builds the CLIP-style vision encoder: a learned patch + position +
    CLS embedding via :class:`OwlViTVisionEmbeddings`, a pre-encoder
    LayerNorm, ``num_hidden_layers`` pre-norm transformer blocks, and
    a post-encoder LayerNorm. The output keeps the CLS token at
    index 0; detection heads in :class:`OwlViTDetect` use the CLS
    token to modulate the patch tokens before the box / class
    predictors.

    Args:
        pixel_values: Image input tensor of shape ``(B, H, W, 3)`` (or
            ``(B, 3, H, W)`` for channels_first).
        hidden_size: Vision tower hidden dimension.
        image_size: Image side length in pixels (square images).
        patch_size: ViT patch size.
        num_hidden_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        block_prefix: Layer name prefix.

    Returns:
        ``(B, num_patches + 1, hidden_size)`` vision encoder output
        (CLS token at index 0 followed by patch tokens).
    """
    x = OwlViTVisionEmbeddings(
        hidden_size=hidden_size,
        image_size=image_size,
        patch_size=patch_size,
        name=f"{block_prefix}_embeddings",
    )(pixel_values)
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{block_prefix}_pre_layernorm")(x)
    x = owlvit_transformer_block(
        x,
        attention_mask=None,
        num_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        block_prefix=block_prefix,
    )
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{block_prefix}_post_layernorm")(
        x
    )
    return x


def owlvit_text_transformer(
    input_ids,
    vocab_size,
    hidden_size,
    max_position_embeddings,
    num_hidden_layers,
    num_heads,
    intermediate_size,
    block_prefix,
):
    """OWL-ViT text tower: token embeddings → causal encoder → LN → EOS pool.

    Builds a CLIP-style text encoder: token + position embeddings via
    :class:`OwlViTTextEmbeddings`, ``num_hidden_layers`` pre-norm
    transformer blocks with a causal attention mask, a final
    LayerNorm, and an EOS-token pooling step. The EOS token is taken
    as the highest-id token in each sequence, so ``argmax(input_ids)``
    gives the pool index per sample.

    Args:
        input_ids: Integer token IDs of shape
            ``(B, max_position_embeddings)``.
        vocab_size: Text tokenizer vocabulary size.
        hidden_size: Text tower hidden dimension.
        max_position_embeddings: Maximum sequence length.
        num_hidden_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        block_prefix: Layer name prefix.

    Returns:
        ``(B, hidden_size)`` EOS-pooled text feature.
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

    x = owlvit_transformer_block(
        x,
        attention_mask=causal,
        num_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        block_prefix=block_prefix,
    )
    x = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_final_layer_norm"
    )(x)

    pool_indices = ops.cast(ops.argmax(input_ids, axis=-1), "int32")
    gather = ops.expand_dims(ops.expand_dims(pool_indices, -1), -1)
    pooled = ops.take_along_axis(x, gather, axis=1)
    pooled = ops.squeeze(pooled, axis=1)
    return pooled


def owlvit_box_predictor(image_features, hidden_size, block_prefix):
    """3-layer MLP predicting raw ``(cx, cy, w, h)`` boxes per patch.

    Applies ``Dense → GELU → Dense → GELU → Dense(4)``. The output is
    a per-patch delta in logit space; :func:`owlvit_detection_head`
    adds an explicit grid bias from :func:`compute_box_bias` and a
    sigmoid so the final boxes lie in normalized ``[0, 1]``
    coordinates.

    Args:
        image_features: Per-patch image features of shape
            ``(B, num_patches, hidden_size)``.
        hidden_size: MLP hidden dimension.
        block_prefix: Layer name prefix.

    Returns:
        ``(B, num_patches, 4)`` raw box predictions (no sigmoid).
    """
    x = layers.Dense(hidden_size, name=f"{block_prefix}_dense0")(image_features)
    x = ops.gelu(x, approximate=False)
    x = layers.Dense(hidden_size, name=f"{block_prefix}_dense1")(x)
    x = ops.gelu(x, approximate=False)
    x = layers.Dense(4, name=f"{block_prefix}_dense2")(x)
    return x


def owlvit_class_predictor(
    image_embeds, query_embeds, query_mask, out_dim, block_prefix
):
    """Text-conditional class predictor with L2-normalized cosine similarity.

    Projects image patch embeddings to ``out_dim``, L2-normalizes both
    the image projections and the per-sample text query embeddings,
    then computes cosine similarity. Two image-conditional scalars
    (a logit shift and a logit scale) are predicted to allow per-image
    temperature calibration. Padding queries
    (``query_mask[..., 0] == 0``) are masked to a very negative logit
    so they don't contribute after softmax.

    Args:
        image_embeds: Image patch embeddings
            ``(B, num_patches, hidden_size)``.
        query_embeds: Per-sample text query embeddings
            ``(B, num_queries, projection_dim)``.
        query_mask: Optional boolean mask of valid queries with shape
            ``(B, num_queries)``. ``True`` keeps the query, ``False``
            masks it out.
        out_dim: Output dimension of the image projection (matches
            ``query_embeds`` last-dim modulo cosine similarity).
        block_prefix: Layer name prefix.

    Returns:
        pred_logits: Cosine-similarity-based class logits
            ``(B, num_patches, num_queries)``.
        image_class_embeds: L2-normalized image projections
            ``(B, num_patches, out_dim)`` — matches HuggingFace's
            returned ``class_embeds`` so callers can use them directly
            for cosine-similarity scoring against custom text queries
            (e.g. one-shot / image-guided detection).
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

    return pred_logits, image_class_embeds_n


def owlvit_functional(
    pixel_values,
    input_ids,
    *,
    vision_image_size,
    vision_patch_size,
    vision_hidden_size,
    vision_intermediate_size,
    vision_num_hidden_layers,
    vision_num_attention_heads,
    text_hidden_size,
    text_intermediate_size,
    text_num_attention_heads,
    text_num_hidden_layers,
    text_max_position_embeddings,
    text_vocab_size,
    projection_dim,
):
    """Build the OWL-ViT dual-tower encoder graph (no detection heads).

    Top-level orchestrator that wires the three encoder stages:

    1. :func:`owlvit_vision_transformer` — CLIP-style ViT vision
       tower producing patch + CLS tokens.
    2. :func:`owlvit_text_transformer` — CLIP-style causal text tower
       with EOS pooling.
    3. A linear text projection to ``projection_dim`` followed by
       L2-normalization, producing the text features used for
       zero-shot class scoring.

    Detection heads (CLS-modulated patch features, box predictor,
    class predictor) are intentionally not built here — they live in
    :func:`owlvit_detection_head` and are added by
    :class:`OwlViTDetect`, which composes :class:`OwlViT` around this
    graph.

    Args:
        pixel_values: Image input tensor.
        input_ids: Text input token IDs.
        vision_image_size: Image side length in pixels.
        vision_patch_size: ViT patch size.
        vision_hidden_size: Vision tower hidden dimension.
        vision_intermediate_size: Vision FFN intermediate dimension.
        vision_num_hidden_layers: Number of vision transformer layers.
        vision_num_attention_heads: Vision attention heads.
        text_hidden_size: Text tower hidden dimension.
        text_intermediate_size: Text FFN intermediate dimension.
        text_num_attention_heads: Text attention heads.
        text_num_hidden_layers: Number of text transformer layers.
        text_max_position_embeddings: Maximum text sequence length.
        text_vocab_size: Tokenizer vocabulary size.
        projection_dim: Output dimension of the text projection.

    Returns:
        image_embeds_raw: Raw vision encoder output
            ``(B, num_patches + 1, vision_hidden_size)`` (CLS at
            index 0).
        text_embeds: L2-normalized text projection
            ``(B, projection_dim)``.
        text_pooled: Unprojected EOS-pooled text feature
            ``(B, text_hidden_size)`` — useful for downstream consumers
            that want the pre-projection representation.
    """
    image_embeds_raw = owlvit_vision_transformer(
        pixel_values,
        hidden_size=vision_hidden_size,
        image_size=vision_image_size,
        patch_size=vision_patch_size,
        num_hidden_layers=vision_num_hidden_layers,
        num_heads=vision_num_attention_heads,
        intermediate_size=vision_intermediate_size,
        block_prefix="vision_model",
    )

    text_pooled = owlvit_text_transformer(
        input_ids,
        vocab_size=text_vocab_size,
        hidden_size=text_hidden_size,
        max_position_embeddings=text_max_position_embeddings,
        num_hidden_layers=text_num_hidden_layers,
        num_heads=text_num_attention_heads,
        intermediate_size=text_intermediate_size,
        block_prefix="text_model",
    )
    text_embeds = layers.Dense(projection_dim, use_bias=False, name="text_projection")(
        text_pooled
    )
    norm = ops.sqrt(ops.sum(text_embeds * text_embeds, axis=-1, keepdims=True) + 1e-12)
    text_embeds = text_embeds / norm

    return image_embeds_raw, text_embeds, text_pooled


def owlvit_detection_head(
    image_embeds_raw,
    text_embeds,
    input_ids,
    vision_hidden_size,
    text_hidden_size,
    num_patches_h,
    num_patches_w,
):
    """Build OWL-ViT detection heads on top of the dual-tower embeddings.

    Mixes the vision CLS token into the patch tokens (elementwise
    multiplication, then LayerNorm), broadcasts the per-batch text
    queries across image samples via :class:`OwlViTSplitBatchQueries`,
    and applies the per-patch box predictor + text-conditional class
    predictor. The box predictor produces raw box deltas; an explicit
    grid bias from :func:`compute_box_bias` is added before sigmoid so
    final boxes lie in normalized ``[0, 1]`` coordinates.

    Args:
        image_embeds_raw: Raw vision encoder output
            ``(B, num_patches + 1, vision_hidden_size)``, including the
            CLS token at index 0.
        text_embeds: L2-normalized text projection
            ``(B, projection_dim)`` from the text tower.
        input_ids: Original text token IDs — used to build the query
            mask (queries whose first token is 0 are treated as padding).
        vision_hidden_size: Vision tower hidden dimension.
        text_hidden_size: Text tower hidden dimension (also the
            class predictor output dim).
        num_patches_h: Number of patches along height.
        num_patches_w: Number of patches along width.

    Returns:
        Dict with keys:
        - ``logits``: ``(B, num_patches, num_queries)`` class logits.
        - ``pred_boxes``: ``(B, num_patches, 4)`` sigmoid-bounded
          boxes in ``(cx, cy, w, h)``.
        - ``text_embeds``: Per-image broadcast query embeddings.
        - ``image_embeds``: Patch features reshaped to a 2D feature
          map ``(B, num_patches_h, num_patches_w, vision_hidden_size)``.
        - ``class_embeds``: Pre-normalization image class projections.
    """
    num_patches = num_patches_h * num_patches_w

    cls_token = image_embeds_raw[:, :1, :]
    patch_embeds = image_embeds_raw[:, 1:, :] * cls_token
    patch_embeds = layers.LayerNormalization(epsilon=1e-5, name="layer_norm")(
        patch_embeds
    )

    feature_map = ops.reshape(
        patch_embeds,
        (-1, num_patches_h, num_patches_w, vision_hidden_size),
    )
    image_feats = ops.reshape(patch_embeds, (-1, num_patches, vision_hidden_size))

    query_embeds = OwlViTSplitBatchQueries(name="split_text_embeds")(
        text_embeds, patch_embeds
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

    return {
        "logits": pred_logits,
        "pred_boxes": pred_boxes,
        "text_embeds": query_embeds,
        "image_embeds": feature_map,
        "class_embeds": class_embeds,
    }


@keras.saving.register_keras_serializable(package="kerasformers")
class OwlViT(BaseModel):
    """OWL-ViT vision + text encoder (no detection heads).

    Mirrors HuggingFace's ``OwlViTModel``. Returns the raw vision
    encoder output and the L2-normalized text projection — suitable
    for zero-shot similarity scoring or as a backbone for custom heads.
    For full detection, use ``OwlViTDetect``.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)
    """

    BASE_MODEL_CONFIG = OWLVIT_CONFIG
    HF_MODEL_TYPE = "owlvit"

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
        text_num_hidden_layers=12,
        text_max_position_embeddings=16,
        text_vocab_size=49408,
        input_shape=None,
        text_input_shape=None,
        name="OwlViT",
        **kwargs,
    ):
        if input_shape is None:
            if keras.config.image_data_format() == "channels_first":
                input_shape = (3, vision_image_size, vision_image_size)
            else:
                input_shape = (vision_image_size, vision_image_size, 3)
        if text_input_shape is None:
            text_input_shape = (text_max_position_embeddings,)

        pixel_values = layers.Input(shape=input_shape, name="pixel_values")
        input_ids = layers.Input(
            shape=text_input_shape, dtype="int32", name="input_ids"
        )

        image_embeds, text_embeds, _ = owlvit_functional(
            pixel_values,
            input_ids,
            vision_image_size=vision_image_size,
            vision_patch_size=vision_patch_size,
            vision_hidden_size=vision_hidden_size,
            vision_intermediate_size=vision_intermediate_size,
            vision_num_hidden_layers=vision_num_hidden_layers,
            vision_num_attention_heads=vision_num_attention_heads,
            text_hidden_size=text_hidden_size,
            text_intermediate_size=text_intermediate_size,
            text_num_attention_heads=text_num_attention_heads,
            text_num_hidden_layers=text_num_hidden_layers,
            text_max_position_embeddings=text_max_position_embeddings,
            text_vocab_size=text_vocab_size,
            projection_dim=projection_dim,
        )

        outputs = {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
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
        self.text_num_hidden_layers = text_num_hidden_layers
        self.text_max_position_embeddings = text_max_position_embeddings
        self.text_vocab_size = text_vocab_size
        self.projection_dim = projection_dim
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
                "text_num_hidden_layers": self.text_num_hidden_layers,
                "text_max_position_embeddings": self.text_max_position_embeddings,
                "text_vocab_size": self.text_vocab_size,
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

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config["vision_config"]
        tc = hf_config["text_config"]
        return {
            "vision_image_size": vc["image_size"],
            "vision_patch_size": vc["patch_size"],
            "vision_hidden_size": vc["hidden_size"],
            "vision_intermediate_size": vc["intermediate_size"],
            "vision_num_hidden_layers": vc["num_hidden_layers"],
            "vision_num_attention_heads": vc["num_attention_heads"],
            "text_hidden_size": tc["hidden_size"],
            "text_intermediate_size": tc["intermediate_size"],
            "text_num_attention_heads": tc["num_attention_heads"],
            "text_num_hidden_layers": tc["num_hidden_layers"],
            "text_max_position_embeddings": tc["max_position_embeddings"],
            "text_vocab_size": tc["vocab_size"],
            "projection_dim": hf_config["projection_dim"],
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_owlvit_hf_to_keras import transfer_owlvit_encoder_weights

        transfer_owlvit_encoder_weights(keras_model, hf_state_dict)


@keras.saving.register_keras_serializable(package="kerasformers")
class OwlViTDetect(BaseModel):
    """OWL-ViT object detection model (encoder + class/box heads).

    Mirrors HuggingFace's ``OwlViTForObjectDetection``. Produces
    per-patch boxes and text-conditional class similarity logits, so
    the set of detection classes is the set of text queries provided
    at inference time rather than a fixed softmax head.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)
    """

    BASE_MODEL_CONFIG = OWLVIT_CONFIG
    BASE_WEIGHT_CONFIG = OWLVIT_WEIGHTS
    HF_MODEL_TYPE = "owlvit"

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
        text_num_hidden_layers=12,
        text_max_position_embeddings=16,
        text_vocab_size=49408,
        input_shape=None,
        text_input_shape=None,
        name="OwlViTDetect",
        **kwargs,
    ):
        num_patches_h = vision_image_size // vision_patch_size
        num_patches_w = vision_image_size // vision_patch_size

        base = OwlViT(
            vision_image_size=vision_image_size,
            vision_patch_size=vision_patch_size,
            vision_hidden_size=vision_hidden_size,
            vision_intermediate_size=vision_intermediate_size,
            vision_num_hidden_layers=vision_num_hidden_layers,
            vision_num_attention_heads=vision_num_attention_heads,
            text_hidden_size=text_hidden_size,
            text_intermediate_size=text_intermediate_size,
            text_num_attention_heads=text_num_attention_heads,
            text_num_hidden_layers=text_num_hidden_layers,
            text_max_position_embeddings=text_max_position_embeddings,
            text_vocab_size=text_vocab_size,
            projection_dim=projection_dim,
            input_shape=input_shape,
            text_input_shape=text_input_shape,
            name=f"{name}_model",
        )
        image_embeds_raw = base.output["image_embeds"]
        text_embeds = base.output["text_embeds"]
        input_ids = base.input["input_ids"]

        outputs = owlvit_detection_head(
            image_embeds_raw,
            text_embeds,
            input_ids,
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=text_hidden_size,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
        )
        super().__init__(inputs=base.input, outputs=outputs, name=name, **kwargs)

        self.vision_image_size = vision_image_size
        self.vision_patch_size = vision_patch_size
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.text_hidden_size = text_hidden_size
        self.text_intermediate_size = text_intermediate_size
        self.text_num_attention_heads = text_num_attention_heads
        self.text_num_hidden_layers = text_num_hidden_layers
        self.text_max_position_embeddings = text_max_position_embeddings
        self.text_vocab_size = text_vocab_size
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
                "text_num_hidden_layers": self.text_num_hidden_layers,
                "text_max_position_embeddings": self.text_max_position_embeddings,
                "text_vocab_size": self.text_vocab_size,
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

    @classmethod
    def config_from_hf(cls, hf_config):
        return OwlViT.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_owlvit_hf_to_keras import transfer_owlvit_detection_weights

        transfer_owlvit_detection_weights(keras_model, hf_state_dict)
