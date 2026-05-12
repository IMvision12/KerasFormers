import keras
from keras import initializers, layers, ops

from kmodels.base import BaseModel
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import SIGLIP_CONFIG, SIGLIP_WEIGHTS
from .siglip_layers import (
    LogitScaleBias,
    PositionEmbedding,
    PositionIDs,
    Probe,
    SigLIPAttention,
)


def siglip_encoder(
    inputs,
    hidden_dim,
    num_heads,
    intermediate_dim,
    layer_norm_epsilon=1e-6,
    name="encoder_layer",
):
    """
    Creates a SigLIP encoder layer with multi-head self-attention and feed-forward network.

    This function implements a transformer encoder layer following the SigLIP architecture,
    which consists of:
    1. Layer normalization followed by multi-head self-attention with residual connection
    2. Layer normalization followed by feed-forward network with residual connection

    Args:
        inputs: Input tensor of shape (batch_size, sequence_length, hidden_dim).
        hidden_dim (int): Dimension of the hidden/embedding space. Must be divisible by num_heads.
        num_heads (int): Number of attention heads for multi-head self-attention.
        intermediate_dim (int): Dimension of the intermediate layer in the feed-forward network.
        layer_norm_epsilon (float, optional): Epsilon value for layer normalization. Defaults to 1e-6.
        name (str, optional): Base name for the layer components. Defaults to "encoder_layer".

    Returns:
        Tensor: Output tensor of the same shape as inputs (batch_size, sequence_length, hidden_dim).

    Raises:
        ValueError: If hidden_dim is not divisible by num_heads.
    """

    if hidden_dim % num_heads != 0:
        raise ValueError(
            "`hidden_dim` must be divisible by `num_heads`. "
            f"Received: hidden_dim={hidden_dim}, num_heads={num_heads}"
        )

    residual1 = inputs
    x = layers.LayerNormalization(
        epsilon=layer_norm_epsilon, name=f"{name}_layernorm_1"
    )(inputs)

    x = SigLIPAttention(
        num_heads,
        hidden_dim // num_heads,
        combined_qkv=False,
        block_prefix=f"{name}_self_attn",
    )(x)

    x = layers.Add(name=f"{name}_add_1")([residual1, x])

    residual2 = x
    x = layers.LayerNormalization(
        epsilon=layer_norm_epsilon, name=f"{name}_layernorm_2"
    )(x)

    x = layers.Dense(
        intermediate_dim,
        bias_initializer=initializers.RandomNormal(stddev=1e-6),
        name=f"{name}_dense_1",
    )(x)
    x = keras.activations.gelu(x, approximate=True)

    x = layers.Dense(
        hidden_dim,
        bias_initializer=initializers.RandomNormal(stddev=1e-6),
        name=f"{name}_dense_2",
    )(x)

    outputs = layers.Add(name=f"{name}_add_2")([residual2, x])

    return outputs


def siglip_attention_pooling(
    inputs,
    hidden_dim,
    intermediate_dim,
    num_heads,
    layer_norm_epsilon=1e-6,
    name="attention_pooling",
):
    """
    Creates a SigLIP attention pooling layer for sequence aggregation.

    This function implements an attention-based pooling mechanism that aggregates
    a sequence of tokens into a single representation. The process involves:
    1. Creating learnable probe tokens
    2. Cross-attention between probes (queries) and input sequence (keys/values)
    3. Feed-forward network with residual connection
    4. Extracting the first token as the final pooled representation

    Args:
        inputs: Input tensor of shape (batch_size, sequence_length, hidden_dim).
        hidden_dim (int): Dimension of the hidden/embedding space.
        intermediate_dim (int): Dimension of the intermediate layer in the feed-forward network.
        num_heads (int): Number of attention heads for multi-head cross-attention.
        layer_norm_epsilon (float, optional): Epsilon value for layer normalization. Defaults to 1e-6.
        name (str, optional): Base name for the layer components. Defaults to "attention_pooling".

    Returns:
        Tensor: Pooled representation of shape (batch_size, hidden_dim). This is the first
               token from the processed probe sequence, representing the aggregated information
               from the entire input sequence.

    Note:
        The function uses cross-attention where the probe tokens act as queries, and the
        input sequence provides both keys and values. The `combined_qkv=True` parameter
        indicates that the attention mechanism uses a combined query-key-value projection.
    """
    probe_layer = Probe(hidden_dim, name=f"{name}_probe")
    probes = probe_layer(inputs)

    hidden_states = SigLIPAttention(
        num_heads,
        hidden_dim // num_heads,
        combined_qkv=True,
        block_prefix=f"{name}_attention",
    )(probes, key=inputs, value=inputs)

    residuals = hidden_states
    x = layers.LayerNormalization(epsilon=layer_norm_epsilon, name=f"{name}_layernorm")(
        hidden_states
    )

    x = layers.Dense(
        intermediate_dim,
        bias_initializer=initializers.RandomNormal(stddev=1e-6),
        name=f"{name}_dense_1",
    )(x)
    x = keras.activations.gelu(x, approximate=True)

    x = layers.Dense(
        hidden_dim,
        bias_initializer=initializers.RandomNormal(stddev=1e-6),
        name=f"{name}_dense_2",
    )(x)

    x = layers.Add(name=f"{name}_add")([residuals, x])

    outputs = x[:, 0]
    return outputs


def siglip_vision_embedding(
    inputs,
    hidden_dim,
    patch_size,
    image_size,
    data_format=None,
    name="vision_embedding",
):
    """
    Creates vision embeddings for SigLIP by converting image patches to embeddings.

    This function implements the vision embedding layer for SigLIP, which transforms
    input images into patch embeddings with positional information. The process involves:
    1. Dividing the image into non-overlapping patches using 2D convolution
    2. Flattening patch embeddings into a sequence
    3. Adding learnable positional embeddings to encode spatial relationships

    Args:
        inputs: Input image tensor. Shape depends on data_format:
               - If data_format="channels_last": (batch_size, height, width, channels)
               - If data_format="channels_first": (batch_size, channels, height, width)
        hidden_dim (int): Dimension of the embedding space for each patch.
        patch_size (int): Size of each square patch. The image is divided into
                         (image_size // patch_size)² patches.
        image_size (int): Size of the input image (assumed to be square).
                         Must be divisible by patch_size.
        data_format (str, optional): Data format for the input tensor.
                                   Either "channels_last" or "channels_first".
                                   If None, uses the default Keras data format.
        name (str, optional): Base name for the layer components.
                             Defaults to "vision_embedding".

    Returns:
        Tensor: Patch embeddings with positional encoding of shape
               (batch_size, num_patches, hidden_dim), where
               num_patches = (image_size // patch_size)².

    Note:
        The patch embedding is performed using a 2D convolution with kernel size
        and stride equal to patch_size, effectively treating each patch as a single
        "pixel" in the output feature map. The LeCun normal initialization is used
        for the convolutional weights.

        Position embeddings use random normal initialization with standard deviation
        scaled by 1/sqrt(hidden_dim) to maintain appropriate variance.
    """

    num_positions = (image_size // patch_size) ** 2
    num_patches_per_side = image_size // patch_size

    patch_embeddings = layers.Conv2D(
        hidden_dim,
        kernel_size=patch_size,
        strides=patch_size,
        kernel_initializer=initializers.LecunNormal(),
        data_format=data_format,
        name=f"{name}_patch_embedding_conv",
    )(inputs)

    if data_format == "channels_last":
        patch_embeddings = layers.Reshape(
            (-1, hidden_dim),
        )(patch_embeddings)
    else:
        patch_embeddings = layers.Reshape(
            (hidden_dim, -1),
        )(patch_embeddings)
        patch_embeddings = layers.Permute(
            (2, 1),
        )(patch_embeddings)

    position_ids = PositionIDs(
        grid_h=num_patches_per_side,
        grid_w=num_patches_per_side,
        use_2d_positions=False,
        name=f"{name}_position_ids",
    )(inputs)

    position_embeddings = PositionEmbedding(
        max_positions=num_positions,
        embedding_dim=hidden_dim,
        embeddings_initializer=initializers.RandomNormal(
            stddev=1.0 / ops.sqrt(hidden_dim)
        ),
        name=f"{name}_position_embedding",
    )(position_ids)

    outputs = layers.Add(name=f"{name}_add_embeddings")(
        [patch_embeddings, position_embeddings]
    )

    return outputs


def siglip_vision_features(
    inputs,
    patch_size,
    hidden_dim,
    num_layers,
    num_heads,
    intermediate_dim,
    layer_norm_epsilon=1e-6,
    data_format=None,
):
    """Pre-pool SigLIP vision encoder output.

    Returns the full token sequence ``(B, num_patches, hidden_dim)`` after
    the post-LN, before any pooling. Matches HF
    ``SiglipVisionModel.last_hidden_state``.
    """
    input_shape = inputs.shape
    if data_format == "channels_last":
        height, width = input_shape[1], input_shape[2]
    else:
        height, width = input_shape[2], input_shape[3]

    if height != width:
        raise ValueError(
            "`siglip_vision_features` expects the height and width to be the "
            f"same in input shape. Received: input_shape={input_shape}"
        )

    x = siglip_vision_embedding(
        inputs,
        hidden_dim=hidden_dim,
        patch_size=patch_size,
        image_size=height,
        data_format=data_format,
        name="vision_model_embeddings",
    )
    for i in range(num_layers):
        x = siglip_encoder(
            x,
            hidden_dim,
            num_heads,
            intermediate_dim,
            layer_norm_epsilon=layer_norm_epsilon,
            name=f"vision_model_encoder_layers_{i}",
        )
    return layers.LayerNormalization(
        epsilon=layer_norm_epsilon, name="vision_model_final_layernorm"
    )(x)


def siglip_vision_encoder(
    inputs,
    patch_size,
    hidden_dim,
    num_layers,
    num_heads,
    intermediate_dim,
    layer_norm_epsilon=1e-6,
    data_format=None,
):
    """Full SigLIP vision encoder: features + attention pooling.

    Returns the pooled vector ``(B, hidden_dim)`` ready for the
    contrastive head. Matches HF ``SiglipVisionModel.pooler_output``.
    """
    x = siglip_vision_features(
        inputs,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        layer_norm_epsilon=layer_norm_epsilon,
        data_format=data_format,
    )
    return siglip_attention_pooling(
        x,
        hidden_dim,
        intermediate_dim,
        num_heads,
        layer_norm_epsilon,
        name="vision_model_head",
    )


def siglip_text_embedding(
    inputs,
    vocabulary_size,
    sequence_length,
    embedding_dim,
    embeddings_initializer="normal",
    mask_zero=False,
    name="text_embedding",
):
    """
    Creates text embeddings for SigLIP by combining token and positional embeddings.

    This function implements the text embedding layer for SigLIP, which transforms
    input token sequences into embeddings with positional information. The process involves:
    1. Converting token IDs to dense token embeddings
    2. Adding learnable positional embeddings to encode sequence order
    3. Combining both embeddings element-wise

    Args:
        inputs: Input token tensor of shape (batch_size, sequence_length) containing
               token IDs from the vocabulary.
        vocabulary_size (int): Size of the token vocabulary. Must be larger than
                              the maximum token ID in the input.
        sequence_length (int): Maximum sequence length for positional embeddings.
                              Should match or exceed the actual sequence length.
        embedding_dim (int): Dimension of the embedding space for both token
                           and positional embeddings.
        embeddings_initializer (str, optional): Initializer for embedding weights.
                                              Defaults to "normal".
        mask_zero (bool, optional): Whether to mask zero values in embeddings.
                                   Useful for variable-length sequences. Defaults to False.
        name (str, optional): Base name for the layer components.
                             Defaults to "text_embedding".

    Returns:
        Tensor: Combined token and positional embeddings of shape
               (batch_size, sequence_length, embedding_dim).
    """
    embedded_tokens = layers.Embedding(
        vocabulary_size,
        embedding_dim,
        embeddings_initializer=embeddings_initializer,
        mask_zero=mask_zero,
        name=f"{name}_token_embedding",
    )(inputs)

    position_ids = PositionIDs(
        grid_h=1,
        grid_w=sequence_length,
        use_2d_positions=False,
        name=f"{name}_position_ids",
    )(inputs)

    embedded_positions = PositionEmbedding(
        max_positions=sequence_length,
        embedding_dim=embedding_dim,
        embeddings_initializer=embeddings_initializer,
        name=f"{name}_position_embedding",
    )(position_ids)

    outputs = layers.Add(name=f"{name}_add_embeddings")(
        [embedded_tokens, embedded_positions]
    )

    return outputs


def siglip_text_encoder(
    inputs,
    vocabulary_size,
    embedding_dim,
    hidden_dim,
    num_layers,
    num_heads,
    intermediate_dim,
    layer_norm_epsilon=1e-6,
    max_sequence_length=64,
    projection_dim=None,
):
    """
    Creates a complete SigLIP text encoder for processing text sequences.

    This function implements the full SigLIP text encoder pipeline that transforms
    input text sequences into dense textual representations. The architecture consists of:
    1. Text embedding layer (token + positional embeddings)
    2. Stack of transformer encoder layers with self-attention
    3. Final layer normalization
    4. Last token extraction and projection to final dimension

    Args:
        inputs: Input token tensor of shape (batch_size, sequence_length) containing
               token IDs from the vocabulary.
        vocabulary_size (int): Size of the token vocabulary.
        embedding_dim (int): Dimension of the input token embeddings.
        hidden_dim (int): Dimension of the hidden/embedding space in transformer layers.
                         Must be divisible by num_heads.
        num_layers (int): Number of transformer encoder layers to stack.
        num_heads (int): Number of attention heads in each transformer layer.
        intermediate_dim (int): Dimension of the feed-forward intermediate layer
                               in each transformer block.
        layer_norm_epsilon (float, optional): Epsilon value for layer normalization.
                                            Defaults to 1e-6.
        max_sequence_length (int, optional): Maximum sequence length for positional
                                           embeddings. Defaults to 64.
        projection_dim (int, optional): Dimension of the final projection layer.
                                      If None, uses hidden_dim. Defaults to None.

    Returns:
        Tensor: Dense textual representation of shape (batch_size, projection_dim).
               This is a single vector per text sequence that encodes the textual content
               suitable for multimodal tasks like vision-language matching.

    Note:
        The encoder uses the last token of the sequence as the final representation,
        which is then projected to the desired output dimension. This approach assumes
        that the last token (often a special [EOS] or [CLS] token) contains the
        most comprehensive sequence-level information.

        The LeCun normal initialization is used for the final projection layer to
        maintain appropriate gradient flow.
    """
    projection_dim = projection_dim or hidden_dim

    x = siglip_text_embedding(
        inputs,
        vocabulary_size=vocabulary_size,
        sequence_length=max_sequence_length,
        embedding_dim=embedding_dim,
        name="text_model_embeddings",
    )

    for i in range(num_layers):
        x = siglip_encoder(
            x,
            hidden_dim,
            num_heads,
            intermediate_dim,
            layer_norm_epsilon=layer_norm_epsilon,
            name=f"text_model_encoder_layers_{i}",
        )

    x = layers.LayerNormalization(
        epsilon=layer_norm_epsilon,
        name="text_model_final_layernorm",
    )(x)

    x = x[:, -1, :]
    outputs = layers.Dense(
        projection_dim,
        kernel_initializer=initializers.LecunNormal(),
        name="text_model_head",
    )(x)

    return outputs


def siglip_head(vision_embedding, text_embedding):
    """
    Computes vision-text similarity logits for SigLIP contrastive learning.

    This function implements the SigLIP head that computes similarity scores between
    vision and text embeddings for contrastive learning. The process involves:
    1. L2 normalization of both vision and text embeddings
    2. Computing cosine similarity matrix between normalized embeddings
    3. Applying learnable logit scale and bias transformation
    4. Returning logits for both vision-to-text and text-to-vision directions

    Args:
        vision_embedding: Vision embeddings tensor of shape (batch_size, embedding_dim).
                         Typically output from a vision encoder (e.g., image features).
        text_embedding: Text embeddings tensor of shape (batch_size, embedding_dim).
                       Typically output from a text encoder (e.g., text features).

    Returns:
        tuple: A tuple containing:
            - image_logits: Tensor of shape (batch_size, batch_size) representing
                           similarity scores from vision to text perspective.
            - text_logits: Tensor of shape (batch_size, batch_size) representing
                          similarity scores from text to vision perspective.

    Note:
        The similarity matrix is computed as cosine similarity between L2-normalized
        embeddings. The LogitScaleBias layer applies learnable scaling and bias
        parameters to the similarity scores, which is crucial for contrastive learning
        optimization in SigLIP.

        The diagonal elements of the logit matrices represent positive pairs
        (matching vision-text pairs), while off-diagonal elements represent
        negative pairs (non-matching pairs).
    """
    vision_norms = ops.sqrt(
        ops.sum(ops.power(vision_embedding, 2), axis=-1, keepdims=True)
    )
    text_norms = ops.sqrt(ops.sum(ops.power(text_embedding, 2), axis=-1, keepdims=True))
    norm_vision = ops.divide(vision_embedding, vision_norms)
    norm_text = ops.divide(text_embedding, text_norms)

    similarity_matrix = ops.matmul(norm_text, ops.transpose(norm_vision))

    text_logits = LogitScaleBias()(similarity_matrix)
    image_logits = ops.transpose(text_logits)

    return image_logits, text_logits


def _siglip_resolve_image_shape(input_shape, image_resolution, data_format):
    if input_shape is not None:
        if data_format == "channels_first":
            channels = input_shape[0] if len(input_shape) == 3 else 3
            image_size = (
                min(input_shape[1], input_shape[2])
                if len(input_shape) == 3
                else input_shape[0]
            )
        else:
            if len(input_shape) >= 2:
                image_size = min(input_shape[0], input_shape[1])
            else:
                image_size = input_shape[0]
            channels = input_shape[2] if len(input_shape) == 3 else 3
    else:
        image_size, channels = image_resolution, 3
    return (
        [channels, image_size, image_size]
        if data_format == "channels_first"
        else [image_size, image_size, channels]
    ), image_size


@keras.saving.register_keras_serializable(package="kmodels")
class SigLIPModel(BaseModel):
    """SigLIP / SigLIP2 dual encoder (no contrastive head).

    Returns the projected vision + text embeddings; use
    :class:`SigLIPZeroShotClassify` for the standard sigmoid-similarity
    head, or :class:`SigLIPImageClassify` for supervised classification.

    Output dict:

    .. code-block:: python

        out = model({"images": ..., "token_ids": ...})
        out["image_embeddings"]   # (B, vision_hidden_dim)
        out["text_embeddings"]    # (B, embed_dim)

    Construction:

    >>> SigLIPModel.from_weights("siglip_base_p16_224")
    >>> SigLIPModel.from_weights("hf:google/siglip-base-patch16-224")
    """

    KMODELS_CONFIG = SIGLIP_CONFIG
    KMODELS_WEIGHTS = SIGLIP_WEIGHTS
    HF_MODEL_TYPE = "siglip"

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config["vision_config"]
        tc = hf_config["text_config"]
        return {
            "image_resolution": vc.get("image_size", 224),
            "patch_size": vc.get("patch_size", 16),
            "vision_hidden_dim": vc.get("hidden_size", 768),
            "vision_num_layers": vc.get("num_hidden_layers", 12),
            "vision_num_heads": vc.get("num_attention_heads", 12),
            "vision_intermediate_dim": vc.get("intermediate_size", 3072),
            "vocabulary_size": tc.get("vocab_size", 32000),
            "embed_dim": tc.get("hidden_size", 768),
            "text_hidden_dim": tc.get("hidden_size", 768),
            "text_num_layers": tc.get("num_hidden_layers", 12),
            "text_num_heads": tc.get("num_attention_heads", 12),
            "text_intermediate_dim": tc.get("intermediate_size", 3072),
            "max_sequence_length": tc.get("max_position_embeddings", 64),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kmodels.models.siglip.convert_siglip_torch_to_keras import (
            transfer_siglip_weights,
        )

        transfer_siglip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        image_resolution=224,
        patch_size=16,
        vision_hidden_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vision_intermediate_dim=3072,
        vocabulary_size=32000,
        embed_dim=768,
        text_hidden_dim=768,
        text_num_layers=12,
        text_num_heads=12,
        text_intermediate_dim=3072,
        max_sequence_length=64,
        input_shape=None,
        input_tensor=None,
        name="SigLIPModel",
        **kwargs,
    ):
        data_format = keras.backend.image_data_format()
        image_input_shape, image_size = _siglip_resolve_image_shape(
            input_shape, image_resolution, data_format
        )

        if isinstance(input_tensor, dict):
            images_input = input_tensor.get("images") or layers.Input(
                shape=image_input_shape, name="images"
            )
            token_ids_input = input_tensor.get("token_ids") or layers.Input(
                shape=(None,), name="token_ids"
            )
        else:
            images_input = layers.Input(shape=image_input_shape, name="images")
            token_ids_input = layers.Input(shape=(None,), name="token_ids")

        vision_embeddings = siglip_vision_encoder(
            images_input,
            patch_size=patch_size,
            hidden_dim=vision_hidden_dim,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads,
            intermediate_dim=vision_intermediate_dim,
            data_format=data_format,
        )

        text_embeddings = siglip_text_encoder(
            token_ids_input,
            vocabulary_size=vocabulary_size,
            embedding_dim=embed_dim,
            hidden_dim=text_hidden_dim,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            intermediate_dim=text_intermediate_dim,
            max_sequence_length=max_sequence_length,
        )

        outputs = {
            "image_embeddings": vision_embeddings,
            "text_embeddings": text_embeddings,
        }
        inputs = {"images": images_input, "token_ids": token_ids_input}

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.image_resolution = image_size
        self.patch_size = patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_intermediate_dim = vision_intermediate_dim
        self.vocabulary_size = vocabulary_size
        self.embed_dim = embed_dim
        self.text_hidden_dim = text_hidden_dim
        self.text_num_layers = text_num_layers
        self.text_num_heads = text_num_heads
        self.text_intermediate_dim = text_intermediate_dim
        self.max_sequence_length = max_sequence_length
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        image_shape_with_batch = self.input_shape["images"]
        if image_shape_with_batch[0] is None:
            image_input_shape = image_shape_with_batch[1:]
        else:
            image_input_shape = image_shape_with_batch
        config.update(
            {
                "image_resolution": self.image_resolution,
                "input_shape": image_input_shape,
                "patch_size": self.patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_intermediate_dim": self.vision_intermediate_dim,
                "vocabulary_size": self.vocabulary_size,
                "embed_dim": self.embed_dim,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_layers": self.text_num_layers,
                "text_num_heads": self.text_num_heads,
                "text_intermediate_dim": self.text_intermediate_dim,
                "max_sequence_length": self.max_sequence_length,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class SigLIPZeroShotClassify(BaseModel):
    """SigLIP + sigmoid-similarity head for zero-shot classification / retrieval.

    Composes the same vision + text encoders as :class:`SigLIPModel` and
    adds the standard SigLIP head — L2-normalize both sides, then a
    learnable ``logit_scale`` and ``logit_bias``. Output is the
    ``(B, B)`` image-vs-text similarity logits.

    >>> SigLIPZeroShotClassify.from_weights("siglip_base_p16_224")
    >>> SigLIPZeroShotClassify.from_weights("hf:google/siglip-base-patch16-224")
    """

    KMODELS_CONFIG = SIGLIP_CONFIG
    KMODELS_WEIGHTS = SIGLIP_WEIGHTS
    HF_MODEL_TYPE = "siglip"

    @classmethod
    def config_from_hf(cls, hf_config):
        return SigLIPModel.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kmodels.models.siglip.convert_siglip_torch_to_keras import (
            transfer_siglip_weights,
        )

        transfer_siglip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        image_resolution=224,
        patch_size=16,
        vision_hidden_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vision_intermediate_dim=3072,
        vocabulary_size=32000,
        embed_dim=768,
        text_hidden_dim=768,
        text_num_layers=12,
        text_num_heads=12,
        text_intermediate_dim=3072,
        max_sequence_length=64,
        input_shape=None,
        input_tensor=None,
        name="SigLIPZeroShotClassify",
        **kwargs,
    ):
        base = SigLIPModel(
            image_resolution=image_resolution,
            patch_size=patch_size,
            vision_hidden_dim=vision_hidden_dim,
            vision_num_layers=vision_num_layers,
            vision_num_heads=vision_num_heads,
            vision_intermediate_dim=vision_intermediate_dim,
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            text_hidden_dim=text_hidden_dim,
            text_num_layers=text_num_layers,
            text_num_heads=text_num_heads,
            text_intermediate_dim=text_intermediate_dim,
            max_sequence_length=max_sequence_length,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_base",
        )
        image_logits, text_logits = siglip_head(
            base.output["image_embeddings"], base.output["text_embeddings"]
        )

        super().__init__(
            inputs=base.input,
            outputs={"image_logits": image_logits, "text_logits": text_logits},
            name=name,
            **kwargs,
        )

        self.image_resolution = image_resolution
        self.patch_size = patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_intermediate_dim = vision_intermediate_dim
        self.vocabulary_size = vocabulary_size
        self.embed_dim = embed_dim
        self.text_hidden_dim = text_hidden_dim
        self.text_num_layers = text_num_layers
        self.text_num_heads = text_num_heads
        self.text_intermediate_dim = text_intermediate_dim
        self.max_sequence_length = max_sequence_length
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        image_shape_with_batch = self.input_shape["images"]
        if image_shape_with_batch[0] is None:
            image_input_shape = image_shape_with_batch[1:]
        else:
            image_input_shape = image_shape_with_batch
        config.update(
            {
                "image_resolution": self.image_resolution,
                "input_shape": image_input_shape,
                "patch_size": self.patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_intermediate_dim": self.vision_intermediate_dim,
                "vocabulary_size": self.vocabulary_size,
                "embed_dim": self.embed_dim,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_layers": self.text_num_layers,
                "text_num_heads": self.text_num_heads,
                "text_intermediate_dim": self.text_intermediate_dim,
                "max_sequence_length": self.max_sequence_length,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class SigLIPImageClassify(BaseModel):
    """SigLIP vision encoder + linear image-classification head.

    Mirrors HF's ``SiglipForImageClassification``: uses **only the SigLIP
    vision encoder** (no text encoder, no attention pooling, no
    ``logit_scale`` / ``logit_bias``), mean-pools the patch tokens, and
    applies a single linear classifier producing ``num_labels`` logits.
    """

    KMODELS_CONFIG = SIGLIP_CONFIG
    KMODELS_WEIGHTS = SIGLIP_WEIGHTS
    HF_MODEL_TYPE = "siglip"

    @classmethod
    def _release_warm_start_cls(cls):
        """Base model class to warm-start the vision encoder from.

        Subclasses (e.g. :class:`SigLIP2ImageClassify`) override this to
        point at their matching encoder-only model.
        """
        return SigLIPModel

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = cls._release_warm_start_cls().from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        from kmodels.base.base_model import hf_num_labels

        config = SigLIPModel.config_from_hf(hf_config)
        try:
            config["num_labels"] = hf_num_labels(hf_config)
        except KeyError:
            pass
        return config

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kmodels.models.siglip.convert_siglip_torch_to_keras import (
            transfer_siglip_image_classify_weights,
        )

        transfer_siglip_image_classify_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        num_labels=1000,
        image_resolution=224,
        patch_size=16,
        vision_hidden_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vision_intermediate_dim=3072,
        input_shape=None,
        input_tensor=None,
        name="SigLIPImageClassify",
        **kwargs,
    ):
        for k in (
            "vocabulary_size",
            "embed_dim",
            "text_hidden_dim",
            "text_num_layers",
            "text_num_heads",
            "text_intermediate_dim",
            "max_sequence_length",
        ):
            kwargs.pop(k, None)

        data_format = keras.backend.image_data_format()
        image_input_shape, image_size = _siglip_resolve_image_shape(
            input_shape, image_resolution, data_format
        )

        if input_tensor is None:
            images_input = layers.Input(shape=image_input_shape, name="images")
        else:
            images_input = input_tensor

        encoded = siglip_vision_features(
            images_input,
            patch_size=patch_size,
            hidden_dim=vision_hidden_dim,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads,
            intermediate_dim=vision_intermediate_dim,
            data_format=data_format,
        )
        pooled = layers.GlobalAveragePooling1D(name="patch_pool")(encoded)
        logits = layers.Dense(num_labels, name="classifier")(pooled)

        super().__init__(inputs=images_input, outputs=logits, name=name, **kwargs)

        self.num_labels = num_labels
        self.image_resolution = image_size
        self.patch_size = patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_intermediate_dim = vision_intermediate_dim
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        image_shape_with_batch = self.input_shape
        if image_shape_with_batch[0] is None:
            image_input_shape = image_shape_with_batch[1:]
        else:
            image_input_shape = image_shape_with_batch
        config.update(
            {
                "num_labels": self.num_labels,
                "image_resolution": self.image_resolution,
                "input_shape": image_input_shape,
                "patch_size": self.patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_intermediate_dim": self.vision_intermediate_dim,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
