import keras
from keras import layers, ops

from kerasformers.base import BaseModel
from kerasformers.utils import standardize_input_shape
from kerasformers.weight_utils import copy_weights_by_path_suffix

from .clip_layers import (
    CLIPAttention,
    CLIPLogitScale,
    CLIPTextModelEmbedding,
    CLIPVisionModelEmbedding,
)
from .config import CLIP_CONFIG, CLIP_WEIGHTS


def quick_gelu(x):
    """Quick GELU approximation used by the original OpenAI CLIP checkpoints.

    Computes ``x * sigmoid(1.702 * x)`` — a cheaper, looser variant of
    GELU. OpenAI's released ``ViT-B/16`` and ``ViT-L/14`` weights were
    trained with this activation, so matching them at inference time
    requires using it here. LAION / community retrains typically use
    standard ``gelu`` instead; switch via the ``hidden_act`` argument
    on the model.

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    return x * ops.sigmoid(1.702 * x)


def activation_layer(hidden_act):
    """Build the activation layer named in the HF CLIP config.

    Recognizes ``"quick_gelu"`` and wraps :func:`quick_gelu` in a
    ``Lambda`` (since it is not registered as a Keras activation).
    Any other name falls through to ``keras.layers.Activation``, which
    handles standard activations like ``"gelu"``, ``"gelu_new"``, or
    ``"relu"``.

    Args:
        hidden_act: Activation name matching HF's ``hidden_act`` field.

    Returns:
        A ``keras.layers.Layer`` instance ready to apply to a tensor.
    """
    if hidden_act == "quick_gelu":
        return keras.layers.Lambda(quick_gelu)
    return keras.layers.Activation(hidden_act)


def residual_attention_block(
    x,
    proj_dim,
    num_heads,
    layer_name_prefix,
    layer_idx,
    causal_attention_mask=None,
    attention_mask=None,
    mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
):
    """One pre-LN residual transformer block (LN → MHSA → Add → LN → MLP → Add).

    Shared building block for both CLIP's vision and text encoders. All
    sublayer names are deterministic — ``{layer_name_prefix}_{layer_idx}_*``
    — so the corresponding PyTorch / HF weights can be transferred by
    name during checkpoint conversion.

    Args:
        x: Input token sequence of shape ``(B, L, proj_dim)``.
        proj_dim: Hidden / model dimension.
        num_heads: Attention head count. ``proj_dim`` must be divisible
            by ``num_heads``.
        layer_name_prefix: Prefix used for every sublayer name (e.g.
            ``"vision_model_encoder"`` or ``"text_model_encoder"``).
        layer_idx: Index of this block within its encoder stack.
        causal_attention_mask: Optional ``(L, L)`` upper-triangular mask
            with large-negative values on disallowed positions, added
            to the attention logits. ``None`` means bidirectional
            attention (vision side).
        attention_mask: Optional broadcastable padding mask. Combined
            additively with ``causal_attention_mask`` when both are
            provided.
        mlp_ratio: MLP hidden expansion ratio — the intermediate Dense
            size is ``int(proj_dim * mlp_ratio)``.
        hidden_act: Activation name for the MLP. Defaults to
            ``"quick_gelu"``.
        layer_norm_eps: Epsilon for both pre-norm LayerNorms.

    Returns:
        Output tensor of shape ``(B, L, proj_dim)``.
    """
    layer_prefix = f"{layer_name_prefix}_{layer_idx}"

    ln_1_output = keras.layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{layer_prefix}_layernorm_1"
    )(x)

    mask = None
    if causal_attention_mask is not None:
        mask = ops.cast(causal_attention_mask, dtype=x.dtype)
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, dtype=x.dtype)
        mask = (
            ops.add(causal_attention_mask, attention_mask)
            if causal_attention_mask is not None
            else attention_mask
        )

    attention_output = CLIPAttention(
        proj_dim=proj_dim,
        num_heads=num_heads,
        name_prefix=f"{layer_prefix}_attn",
    )(ln_1_output, attention_mask=mask)[0]

    residual_1 = keras.layers.Add()([x, attention_output])
    ln_2_output = keras.layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{layer_prefix}_layernorm_2"
    )(residual_1)

    mlp_intermediate_size = int(proj_dim * mlp_ratio)
    mlp_output = keras.layers.Dense(
        mlp_intermediate_size, name=f"{layer_prefix}_dense_1"
    )(ln_2_output)
    mlp_output = activation_layer(hidden_act)(mlp_output)
    mlp_output = keras.layers.Dense(proj_dim, name=f"{layer_prefix}_dense_2")(
        mlp_output
    )

    return keras.layers.Add()([residual_1, mlp_output])


def clip_encoder(
    inputs,
    width,
    num_layers,
    heads,
    layer_prefix=None,
    causal_attention_mask=None,
    attention_mask=None,
    mlp_ratio=None,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
):
    """Stack of ``num_layers`` CLIP transformer blocks.

    Threads the running tensor through :func:`residual_attention_block`
    ``num_layers`` times. The same ``layer_prefix`` and a per-iteration
    ``layer_idx`` give every sublayer a unique, stable name
    (e.g. ``vision_model_encoder_3_attn_...``), which is what the weight
    transfer scripts rely on.

    Args:
        inputs: Token sequence of shape ``(B, L, width)``.
        width: Hidden dimension.
        num_layers: Number of transformer blocks to stack.
        heads: Attention head count per block.
        layer_prefix: Block-name prefix shared by every layer in this
            stack (e.g. ``"vision_model_encoder"``).
        causal_attention_mask: Optional ``(L, L)`` causal mask, forwarded
            to each block. Used by the text encoder only.
        attention_mask: Optional padding mask broadcastable over the
            attention logits.
        mlp_ratio: MLP expansion ratio passed to each block.
        hidden_act: MLP activation name. Defaults to ``"quick_gelu"``.
        layer_norm_eps: LayerNorm epsilon used inside every block.

    Returns:
        Tensor of shape ``(B, L, width)``.
    """
    x = inputs
    for i in range(num_layers):
        x = residual_attention_block(
            x,
            proj_dim=width,
            num_heads=heads,
            layer_name_prefix=layer_prefix,
            layer_idx=i,
            causal_attention_mask=causal_attention_mask,
            attention_mask=attention_mask,
            mlp_ratio=mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )
    return x


def clip_vision_features(
    inputs,
    input_resolution=224,
    patch_size=16,
    width=768,
    num_layers=12,
    heads=12,
    vision_mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    data_format="channels_last",
):
    """CLIP vision encoder up through the transformer stack (no projection).

    Pipeline: patch ``Conv2D`` → prepend the learned CLS token and add
    positional embeddings via :class:`CLIPVisionModelEmbedding` → pre-LN →
    :func:`clip_encoder`. Output is the full token sequence (CLS at
    index 0), matching HF's ``CLIPVisionModel.last_hidden_state`` —
    useful when you want raw features rather than the projected image
    embedding.

    Args:
        inputs: Image tensor. Shape is ``(B, H, W, C)`` for
            ``channels_last`` or ``(B, C, H, W)`` for ``channels_first``.
        input_resolution: Image side length, used to size the learned
            positional embeddings.
        patch_size: Square patch side length.
        width: Hidden dimension.
        num_layers: Transformer depth.
        heads: Attention head count.
        vision_mlp_ratio: MLP expansion ratio.
        hidden_act: MLP activation name.
        layer_norm_eps: LayerNorm epsilon.
        data_format: ``"channels_last"`` or ``"channels_first"``.

    Returns:
        Tensor of shape ``(B, num_patches + 1, width)`` — CLS token at
        index 0, followed by patch tokens.
    """
    patch_embeddings = keras.layers.Conv2D(
        filters=width,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="vision_model_conv",
    )(inputs)

    embeddings = CLIPVisionModelEmbedding(
        width, input_resolution, patch_size, data_format, name="vision_model_embeddings"
    )(patch_embeddings)

    x = keras.layers.LayerNormalization(
        epsilon=layer_norm_eps, name="vision_model_layernorm_1"
    )(embeddings)
    return clip_encoder(
        x,
        width=width,
        num_layers=num_layers,
        heads=heads,
        layer_prefix="vision_model_encoder",
        mlp_ratio=vision_mlp_ratio,
        hidden_act=hidden_act,
        layer_norm_eps=layer_norm_eps,
    )


def clip_image_encoder(
    inputs,
    input_resolution=224,
    patch_size=16,
    width=768,
    num_layers=12,
    heads=12,
    output_dim=512,
    vision_mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    data_format="channels_last",
):
    """Full CLIP image encoder: features → CLS → post-LN → visual projection.

    Wraps :func:`clip_vision_features` and finishes the CLIP image side
    — slice the CLS token, apply the post-encoder LayerNorm, then
    project into the shared ``output_dim`` embedding space with a
    bias-free Dense layer (the ``visual_projection`` weight in HF CLIP).
    This is the tensor used by the contrastive head; it is not yet
    L2-normalized.

    Args:
        inputs: Image tensor.
        input_resolution: Image side length.
        patch_size: ViT patch size.
        width: Hidden dimension.
        num_layers: Transformer depth.
        heads: Attention head count.
        output_dim: Target embedding dimension (must match the text
            side's ``embed_dim``).
        vision_mlp_ratio: MLP expansion ratio.
        hidden_act: MLP activation name.
        layer_norm_eps: LayerNorm epsilon.
        data_format: ``"channels_last"`` or ``"channels_first"``.

    Returns:
        Tensor of shape ``(B, output_dim)`` — the unnormalized image
        embedding.
    """
    encoded = clip_vision_features(
        inputs,
        input_resolution=input_resolution,
        patch_size=patch_size,
        width=width,
        num_layers=num_layers,
        heads=heads,
        vision_mlp_ratio=vision_mlp_ratio,
        hidden_act=hidden_act,
        layer_norm_eps=layer_norm_eps,
        data_format=data_format,
    )

    class_token = keras.layers.Lambda(lambda x: x[:, 0, :], name="extract_token")(
        encoded
    )
    x = keras.layers.LayerNormalization(
        epsilon=layer_norm_eps, name="vision_model_layernorm_2"
    )(class_token)
    return keras.layers.Dense(output_dim, use_bias=False, name="visual_projection")(x)


def clip_text_encoder(
    inputs,
    attention_mask,
    transformer_width,
    transformer_layers,
    transformer_heads,
    vocab_size,
    embed_dim,
    context_length,
    text_mlp_ratio,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
):
    """CLIP text encoder with causal attention and EOT-token pooling.

    Pipeline: :class:`CLIPTextModelEmbedding` (token + positional) →
    :func:`clip_encoder` with a strict upper-triangular causal mask
    plus the padding mask → post-encoder LayerNorm → pluck the hidden
    state at each row's EOT position (HF picks the position with the
    largest ``token_id`` — typically the end-of-text token) → text
    projection. Yields one ``(B, embed_dim)`` feature per caption.

    Args:
        inputs: Token-id tensor of shape ``(B, context_length)``.
        attention_mask: Padding mask ``(B, context_length)`` — ``1`` for
            real tokens, ``0`` for padding.
        transformer_width: Text encoder hidden dimension.
        transformer_layers: Text encoder depth.
        transformer_heads: Attention head count.
        vocab_size: Tokenizer vocabulary size.
        embed_dim: Shared joint embedding dimension.
        context_length: Maximum sequence length, used both for the
            causal mask and the positional embedding table.
        text_mlp_ratio: MLP expansion ratio.
        hidden_act: MLP activation name.
        layer_norm_eps: LayerNorm epsilon.

    Returns:
        Tensor of shape ``(B, embed_dim)`` — the unnormalized text
        embedding.
    """
    x = CLIPTextModelEmbedding(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=transformer_width,
        name="text_model_embedding",
    )(inputs)

    causal_attention_mask = ops.cast(
        ops.triu(ops.ones((context_length, context_length)), k=1), "float32"
    ) * (-1e8)

    attention_mask_float = ops.cast(attention_mask, dtype="float32")
    expanded_mask = ops.reshape(attention_mask_float, (-1, 1, 1, context_length))
    expanded_mask = ops.repeat(expanded_mask, context_length, axis=2)
    expanded_mask = (1.0 - expanded_mask) * (-1e8)

    encoded_output = clip_encoder(
        x,
        width=transformer_width,
        num_layers=transformer_layers,
        heads=transformer_heads,
        causal_attention_mask=causal_attention_mask,
        attention_mask=expanded_mask,
        mlp_ratio=text_mlp_ratio,
        layer_prefix="text_model_encoder",
        hidden_act=hidden_act,
        layer_norm_eps=layer_norm_eps,
    )

    layer_norm = keras.layers.LayerNormalization(
        epsilon=layer_norm_eps, name="text_model_layernorm"
    )(encoded_output)

    indices = ops.argmax(inputs, axis=-1)
    one_hot_indices = ops.one_hot(indices, context_length)
    selected_features = ops.einsum("bi,bij->bj", one_hot_indices, layer_norm)
    selected_features = ops.expand_dims(selected_features, axis=1)

    text_features = keras.layers.Dense(
        embed_dim, name="text_projection", use_bias=False
    )(selected_features)

    return ops.squeeze(text_features, axis=1)


def clip_head(image_embeddings, text_embeddings):
    """L2-normalize embeddings and produce scaled similarity logits.

    Standard CLIP contrastive head. Each side is L2-normalized along
    its embedding axis (turning the dot product into cosine similarity),
    then both are passed to :class:`CLIPLogitScale`, which multiplies by
    the learned ``exp(logit_scale)`` temperature and returns the
    ``(B, B)`` image-vs-text similarity logit matrix together with its
    transpose.

    Args:
        image_embeddings: Image embedding tensor ``(B, embed_dim)``.
        text_embeddings: Text embedding tensor ``(B, embed_dim)``.

    Returns:
        Tuple ``(image_logits, text_logits)``, each of shape ``(B, B)``.
        ``image_logits[i, j]`` is the temperature-scaled cosine
        similarity between image ``i`` and text ``j``.
    """
    image_embeddings = image_embeddings / ops.sqrt(
        ops.sum(ops.power(image_embeddings, 2), axis=-1, keepdims=True)
    )
    text_embeddings = text_embeddings / ops.sqrt(
        ops.sum(ops.power(text_embeddings, 2), axis=-1, keepdims=True)
    )
    logit_scale_layer = CLIPLogitScale(initial_value=0.07, name="logit_scale")
    return logit_scale_layer([image_embeddings, text_embeddings])


@keras.saving.register_keras_serializable(package="kerasformers")
class CLIPModel(BaseModel):
    """Contrastive Language-Image Pre-training (CLIP) dual encoder.

    Joint vision + text encoder pair projecting to a shared embedding
    space. Returns the projected embeddings on each side — *no*
    similarity / logit-scale head is applied; use
    :class:`CLIPZeroShotClassify` for the standard contrastive head,
    or call :meth:`CLIPModel` and compute similarity yourself.

    Output dict:

    .. code-block:: python

        out = model({"images": ..., "token_ids": ..., "padding_mask": ...})
        out["image_embeddings"]   # (B, embed_dim)
        out["text_embeddings"]    # (B, embed_dim)

    Construction:

    >>> CLIPModel.from_weights("clip_vit_base_16")             # kerasformers release
    >>> CLIPModel.from_weights("hf:openai/clip-vit-base-patch16")
    >>> CLIPModel.from_weights("hf:laion/CLIP-ViT-B-16-laion2B-s34B-b88K")

    Reference:
        - `Learning Transferable Visual Models From Natural Language
          Supervision <https://arxiv.org/abs/2103.00020>`_

    Args:
        embed_dim: Shared embedding dim (= HF ``projection_dim``).
        input_image_shape: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()`` —
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        vision_layers: ViT encoder depth.
        vision_width: ViT hidden dim.
        vision_patch_size: ViT patch size.
        context_length: Text input length.
        vocab_size: Tokenizer vocab size.
        transformer_width: Text encoder hidden dim.
        transformer_heads: Text encoder head count.
        transformer_layers: Text encoder depth.
        vision_mlp_ratio: MLP expansion ratio in vision blocks.
        text_mlp_ratio: MLP expansion ratio in text blocks.
        hidden_act: MLP activation. ``"quick_gelu"`` for canonical
            OpenAI CLIP; ``"gelu"`` / ``"gelu_new"`` for LAION /
            community variants.
        layer_norm_eps: Epsilon for every LayerNorm. Defaults to ``1e-5``.
        input_tensor: Optional dict of pre-existing input tensors.
        name: Model name.
    """

    BASE_MODEL_CONFIG = CLIP_CONFIG
    BASE_WEIGHT_CONFIG = CLIP_WEIGHTS
    HF_MODEL_TYPE = "clip"

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config["vision_config"]
        tc = hf_config["text_config"]
        return {
            "embed_dim": hf_config["projection_dim"],
            "input_image_shape": vc.get("image_size", 224),
            "vision_layers": vc["num_hidden_layers"],
            "vision_width": vc["hidden_size"],
            "vision_patch_size": vc["patch_size"],
            "context_length": tc.get("max_position_embeddings", 77),
            "vocab_size": tc["vocab_size"],
            "transformer_width": tc["hidden_size"],
            "transformer_heads": tc["num_attention_heads"],
            "transformer_layers": tc["num_hidden_layers"],
            "vision_mlp_ratio": vc["intermediate_size"] / vc["hidden_size"],
            "text_mlp_ratio": tc["intermediate_size"] / tc["hidden_size"],
            "hidden_act": vc.get("hidden_act", "quick_gelu"),
            "layer_norm_eps": vc.get("layer_norm_eps", 1e-5),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_clip_torch_to_keras import transfer_clip_weights

        transfer_clip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        embed_dim=512,
        input_image_shape=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        vision_mlp_ratio=4.0,
        text_mlp_ratio=4.0,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        input_tensor=None,
        name="CLIPModel",
        **kwargs,
    ):
        vision_heads = vision_width // 64
        data_format = keras.config.image_data_format()
        input_image_shape = standardize_input_shape(input_image_shape, data_format)
        if data_format == "channels_first":
            image_size = input_image_shape[1]
        else:
            image_size = input_image_shape[0]

        if isinstance(input_tensor, dict):
            images_input = input_tensor.get("images") or layers.Input(
                shape=input_image_shape, name="images"
            )
            token_ids_input = input_tensor.get("token_ids") or layers.Input(
                shape=[context_length], name="token_ids"
            )
            padding_mask_input = input_tensor.get("padding_mask") or layers.Input(
                shape=[context_length], name="padding_mask"
            )
        else:
            images_input = layers.Input(shape=input_image_shape, name="images")
            token_ids_input = layers.Input(shape=[context_length], name="token_ids")
            padding_mask_input = layers.Input(
                shape=[context_length], name="padding_mask"
            )

        image_embeddings = clip_image_encoder(
            images_input,
            input_resolution=image_size,
            patch_size=vision_patch_size,
            width=vision_width,
            num_layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            data_format=data_format,
        )

        text_embeddings = clip_text_encoder(
            token_ids_input,
            attention_mask=padding_mask_input,
            transformer_width=transformer_width,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            text_mlp_ratio=text_mlp_ratio,
            context_length=context_length,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

        outputs = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }
        inputs = {
            "images": images_input,
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.embed_dim = embed_dim
        self.input_image_shape = input_image_shape
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.vision_mlp_ratio = vision_mlp_ratio
        self.text_mlp_ratio = text_mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "input_image_shape": self.input_image_shape,
                "vision_layers": self.vision_layers,
                "vision_width": self.vision_width,
                "vision_patch_size": self.vision_patch_size,
                "context_length": self.context_length,
                "vocab_size": self.vocab_size,
                "transformer_width": self.transformer_width,
                "transformer_heads": self.transformer_heads,
                "transformer_layers": self.transformer_layers,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "text_mlp_ratio": self.text_mlp_ratio,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class CLIPZeroShotClassify(BaseModel):
    """CLIP + contrastive similarity head for zero-shot classification / retrieval.

    Composes the same vision + text encoders as :class:`CLIPModel` and
    adds the standard CLIP head — L2-normalize both sides, then a
    learnable ``logit_scale`` temperature on the cosine-similarity
    matrix. Output is the ``(B, B)`` image-vs-text similarity logits,
    which softmax to zero-shot class probabilities when ``text_*``
    inputs are class-name prompts.

    Output dict:

    .. code-block:: python

        out = model({"images": ..., "token_ids": ..., "padding_mask": ...})
        out["image_logits"]   # (B, B) — image[i] vs text[j], scaled
        out["text_logits"]    # (B, B) — transpose of image_logits

    Construction:

    >>> CLIPZeroShotClassify.from_weights("clip_vit_base_16")
    >>> CLIPZeroShotClassify.from_weights("hf:openai/clip-vit-base-patch16")

    Args (identical to :class:`CLIPModel`):
        embed_dim, input_image_shape, vision_layers, vision_width,
        vision_patch_size, context_length, vocab_size, transformer_width,
        transformer_heads, transformer_layers, vision_mlp_ratio,
        text_mlp_ratio, hidden_act, layer_norm_eps, input_tensor, name.
    """

    BASE_MODEL_CONFIG = CLIP_CONFIG
    BASE_WEIGHT_CONFIG = CLIP_WEIGHTS
    HF_MODEL_TYPE = "clip"

    @classmethod
    def config_from_hf(cls, hf_config):
        return CLIPModel.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_clip_torch_to_keras import transfer_clip_weights

        transfer_clip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        embed_dim=512,
        input_image_shape=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        vision_mlp_ratio=4.0,
        text_mlp_ratio=4.0,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        input_tensor=None,
        name="CLIPZeroShotClassify",
        **kwargs,
    ):
        base = CLIPModel(
            embed_dim=embed_dim,
            input_image_shape=input_image_shape,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            vision_mlp_ratio=vision_mlp_ratio,
            text_mlp_ratio=text_mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            input_tensor=input_tensor,
            name=f"{name}_base",
        )

        image_embeddings = base.output["image_embeddings"]
        text_embeddings = base.output["text_embeddings"]
        image_logits, text_logits = clip_head(image_embeddings, text_embeddings)

        super().__init__(
            inputs=base.input,
            outputs={"image_logits": image_logits, "text_logits": text_logits},
            name=name,
            **kwargs,
        )

        self.embed_dim = embed_dim
        self.input_image_shape = base.input_image_shape
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.vision_mlp_ratio = vision_mlp_ratio
        self.text_mlp_ratio = text_mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "input_image_shape": self.input_image_shape,
                "vision_layers": self.vision_layers,
                "vision_width": self.vision_width,
                "vision_patch_size": self.vision_patch_size,
                "context_length": self.context_length,
                "vocab_size": self.vocab_size,
                "transformer_width": self.transformer_width,
                "transformer_heads": self.transformer_heads,
                "transformer_layers": self.transformer_layers,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "text_mlp_ratio": self.text_mlp_ratio,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class CLIPImageClassify(BaseModel):
    """CLIP vision encoder + linear image-classification head.

    Mirrors HF's ``CLIPForImageClassification``: uses **only the CLIP
    vision encoder** (no text encoder, no visual projection), then
    mean-pools the patch tokens (excluding CLS) and applies a single
    linear classifier producing ``num_labels`` logits.

    .. code-block:: python

        model = CLIPImageClassify.from_weights(
            "hf:<user>/clip-finetune-imagenet"
        )
        logits = model(images)              # (B, num_labels)

    Args:
        num_labels: Number of output classes.
        input_image_shape: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()`` —
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        vision_layers: ViT encoder depth.
        vision_width: ViT hidden dim.
        vision_patch_size: ViT patch size.
        vision_mlp_ratio: MLP expansion ratio in vision blocks.
        hidden_act: MLP activation. ``"quick_gelu"`` for OpenAI,
            ``"gelu"`` / ``"gelu_new"`` for community variants.
        layer_norm_eps: Epsilon for every LayerNorm. Defaults to ``1e-5``.
        input_tensor: Optional pre-existing input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = CLIP_CONFIG
    BASE_WEIGHT_CONFIG = CLIP_WEIGHTS
    HF_MODEL_TYPE = "clip"

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = CLIPModel.from_weights(variant, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        from kerasformers.base.base_model import hf_num_labels

        config = CLIPModel.config_from_hf(hf_config)
        try:
            config["num_labels"] = hf_num_labels(hf_config)
        except KeyError:
            pass
        return config

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_clip_torch_to_keras import transfer_clip_image_classify_weights

        transfer_clip_image_classify_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        num_labels=1000,
        input_image_shape=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        vision_mlp_ratio=4.0,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        input_tensor=None,
        name="CLIPImageClassify",
        **kwargs,
    ):
        for k in (
            "embed_dim",
            "context_length",
            "vocab_size",
            "transformer_width",
            "transformer_heads",
            "transformer_layers",
            "text_mlp_ratio",
        ):
            kwargs.pop(k, None)

        vision_heads = vision_width // 64
        data_format = keras.config.image_data_format()
        input_image_shape = standardize_input_shape(input_image_shape, data_format)
        if data_format == "channels_first":
            image_size = input_image_shape[1]
        else:
            image_size = input_image_shape[0]

        if input_tensor is None:
            images_input = layers.Input(shape=input_image_shape, name="images")
        else:
            images_input = input_tensor

        encoded = clip_vision_features(
            images_input,
            input_resolution=image_size,
            patch_size=vision_patch_size,
            width=vision_width,
            num_layers=vision_layers,
            heads=vision_heads,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            data_format=data_format,
        )

        pooled = ops.mean(encoded[:, 1:, :], axis=1)
        logits = layers.Dense(num_labels, name="classifier")(pooled)

        super().__init__(inputs=images_input, outputs=logits, name=name, **kwargs)

        self.num_labels = num_labels
        self.input_image_shape = input_image_shape
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.vision_mlp_ratio = vision_mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_labels": self.num_labels,
                "input_image_shape": self.input_image_shape,
                "vision_layers": self.vision_layers,
                "vision_width": self.vision_width,
                "vision_patch_size": self.vision_patch_size,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
