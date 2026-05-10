import keras
from keras import layers, ops


def quick_gelu(x):
    return x * ops.sigmoid(1.702 * x)


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTVisionEmbeddings(layers.Layer):
    """Patch embedding + class token + learned position embedding for OWL-ViT.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        hidden_size: Integer, embedding dimension of each patch token.
        image_size: Integer, square image edge in pixels.
        patch_size: Integer, square patch edge in pixels. Must divide
            ``image_size``.
        num_channels: Integer, number of input image channels. Defaults
            to ``3``.
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Input Shape:
        4D tensor: ``(batch_size, image_size, image_size, num_channels)``.

    Output Shape:
        3D tensor: ``(batch_size, num_patches + 1, hidden_size)``.
    """

    def __init__(
        self,
        hidden_size,
        image_size,
        patch_size,
        num_channels=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.patch_embedding = layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=False,
            data_format="channels_last",
            name="patch_embedding",
        )
        self.position_embedding = layers.Embedding(
            self.num_positions,
            hidden_size,
            name="position_embedding",
        )

    def build(self, input_shape):
        self.class_embedding = self.add_weight(
            name="class_embedding",
            shape=(self.hidden_size,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        b = ops.shape(patch_embeds)[0]
        patch_embeds = ops.reshape(
            patch_embeds, (b, self.num_patches, self.hidden_size)
        )
        cls = ops.broadcast_to(
            ops.reshape(self.class_embedding, (1, 1, self.hidden_size)),
            (b, 1, self.hidden_size),
        )
        embeddings = ops.concatenate([cls, patch_embeds], axis=1)
        position_ids = ops.arange(0, self.num_positions, dtype="int32")
        pos_embed = self.position_embedding(position_ids)
        return embeddings + ops.expand_dims(pos_embed, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_channels": self.num_channels,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTTextEmbeddings(layers.Layer):
    """Token embedding + learned position embedding for the OWL-ViT text tower.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        vocab_size: Integer, text vocabulary size.
        hidden_size: Integer, hidden size of the text tower.
        max_position_embeddings: Integer, maximum text sequence length.
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Input Shape:
        2D integer tensor: ``(batch_size, sequence_length)``.

    Output Shape:
        3D tensor: ``(batch_size, sequence_length, hidden_size)``.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings

        self.token_embedding = layers.Embedding(
            vocab_size,
            hidden_size,
            name="token_embedding",
        )
        self.position_embedding = layers.Embedding(
            max_position_embeddings,
            hidden_size,
            name="position_embedding",
        )

    def call(self, input_ids):
        seq_len = ops.shape(input_ids)[-1]
        token_embeds = self.token_embedding(input_ids)
        position_ids = ops.arange(0, seq_len, dtype="int32")
        position_embeds = self.position_embedding(position_ids)
        return token_embeds + ops.expand_dims(position_embeds, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "max_position_embeddings": self.max_position_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTAttention(layers.Layer):
    """Multi-head self-attention with separate q/k/v/out projections.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        hidden_size: Integer, total model dimension. Must be divisible
            by ``num_heads``.
        num_heads: Integer, number of parallel attention heads.
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Input Shape:
        3D tensor: ``(batch_size, seq_len, hidden_size)``. An optional
        additive ``attention_mask`` broadcastable to
        ``(batch_size, num_heads, seq_len, seq_len)`` may be provided.

    Output Shape:
        3D tensor: ``(batch_size, seq_len, hidden_size)``.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({num_heads})."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = layers.Dense(hidden_size, name="k_proj")
        self.v_proj = layers.Dense(hidden_size, name="v_proj")
        self.q_proj = layers.Dense(hidden_size, name="q_proj")
        self.out_proj = layers.Dense(hidden_size, name="out_proj")

    def _split_heads(self, x):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        x = ops.reshape(x, (b, s, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states, attention_mask=None):
        q = self._split_heads(self.q_proj(hidden_states))
        k = self._split_heads(self.k_proj(hidden_states))
        v = self._split_heads(self.v_proj(hidden_states))

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.softmax(attn, axis=-1)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        b = ops.shape(out)[0]
        s = ops.shape(out)[1]
        out = ops.reshape(out, (b, s, self.hidden_size))
        return self.out_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTMLP(layers.Layer):
    """Two-layer MLP block (``fc1`` → activation → ``fc2``).

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        hidden_size: Integer, model dimension.
        intermediate_size: Integer, MLP expansion dimension.
        hidden_act: String, activation function. ``"quick_gelu"`` matches
            the HF defaults; ``"gelu"`` is also supported.
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Input/Output Shape:
        3D tensor: ``(batch_size, seq_len, hidden_size)``.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_act="quick_gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.fc1 = layers.Dense(intermediate_size, name="fc1")
        self.fc2 = layers.Dense(hidden_size, name="fc2")

    def call(self, hidden_states):
        x = self.fc1(hidden_states)
        if self.hidden_act == "quick_gelu":
            x = quick_gelu(x)
        elif self.hidden_act == "gelu":
            x = ops.gelu(x, approximate=False)
        else:
            x = keras.activations.get(self.hidden_act)(x)
        return self.fc2(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "hidden_act": self.hidden_act,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTEncoderLayer(layers.Layer):
    """Pre-norm transformer block: LN → SA → residual → LN → MLP → residual.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        hidden_size: Integer, model dimension.
        num_heads: Integer, number of attention heads.
        intermediate_size: Integer, MLP expansion dimension.
        layer_norm_eps: Float, layer normalization epsilon. Defaults
            to ``1e-5``.
        hidden_act: String, MLP activation. Defaults to ``"quick_gelu"``.
        **kwargs: Additional keyword arguments passed to ``Layer``.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        layer_norm_eps=1e-5,
        hidden_act="quick_gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

        self.self_attn = OwlViTAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            name="self_attn",
        )
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=layer_norm_eps,
            name="layer_norm1",
        )
        self.mlp = OwlViTMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            name="mlp",
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=layer_norm_eps,
            name="layer_norm2",
        )

    def call(self, hidden_states, attention_mask=None):
        residual = hidden_states
        x = self.layer_norm1(hidden_states)
        x = self.self_attn(x, attention_mask=attention_mask)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        return residual + x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "layer_norm_eps": self.layer_norm_eps,
                "hidden_act": self.hidden_act,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTEncoder(layers.Layer):
    """Stack of ``OwlViTEncoderLayer`` blocks named ``layers_{i}``.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        num_hidden_layers: Integer, number of stacked encoder layers.
        hidden_size: Integer, model dimension.
        num_heads: Integer, number of attention heads per layer.
        intermediate_size: Integer, MLP expansion dimension per layer.
        layer_norm_eps: Float, layer normalization epsilon. Defaults
            to ``1e-5``.
        hidden_act: String, MLP activation. Defaults to ``"quick_gelu"``.
        **kwargs: Additional keyword arguments passed to ``Layer``.
    """

    def __init__(
        self,
        num_hidden_layers,
        hidden_size,
        num_heads,
        intermediate_size,
        layer_norm_eps=1e-5,
        hidden_act="quick_gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

        self.encoder_layers = [
            OwlViTEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                layer_norm_eps=layer_norm_eps,
                hidden_act=hidden_act,
                name=f"layers_{i}",
            )
            for i in range(num_hidden_layers)
        ]

    def call(self, hidden_states, attention_mask=None):
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "layer_norm_eps": self.layer_norm_eps,
                "hidden_act": self.hidden_act,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTVisionTransformer(layers.Layer):
    """OWL-ViT vision tower: embeddings → pre LN → encoder → post LN.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        hidden_size: Integer, model dimension.
        image_size: Integer, square image edge in pixels.
        patch_size: Integer, square patch edge in pixels.
        num_hidden_layers: Integer, number of encoder layers.
        num_heads: Integer, attention heads per layer.
        intermediate_size: Integer, MLP expansion dimension per layer.
        layer_norm_eps: Float, layer normalization epsilon. Defaults
            to ``1e-5``.
        hidden_act: String, MLP activation. Defaults to ``"quick_gelu"``.
        num_channels: Integer, input channels. Defaults to ``3``.
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Input Shape:
        4D tensor: ``(batch_size, image_size, image_size, num_channels)``.

    Output Shape:
        3D tensor: ``(batch_size, num_patches + 1, hidden_size)``.
    """

    def __init__(
        self,
        hidden_size,
        image_size,
        patch_size,
        num_hidden_layers,
        num_heads,
        intermediate_size,
        layer_norm_eps=1e-5,
        hidden_act="quick_gelu",
        num_channels=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.num_channels = num_channels

        self.embeddings = OwlViTVisionEmbeddings(
            hidden_size=hidden_size,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            name="embeddings",
        )
        self.pre_layernorm = layers.LayerNormalization(
            epsilon=layer_norm_eps,
            name="pre_layernorm",
        )
        self.encoder = OwlViTEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
            name="encoder",
        )
        self.post_layernorm = layers.LayerNormalization(
            epsilon=layer_norm_eps,
            name="post_layernorm",
        )

    def call(self, pixel_values):
        x = self.embeddings(pixel_values)
        x = self.pre_layernorm(x)
        x = self.encoder(x, attention_mask=None)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "layer_norm_eps": self.layer_norm_eps,
                "hidden_act": self.hidden_act,
                "num_channels": self.num_channels,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTTextTransformer(layers.Layer):
    """OWL-ViT text tower: embeddings → causal encoder → final LN.

    Returns ``(last_hidden_state, pooled_output)`` where the pooled
    output is gathered at the per-row argmax of ``input_ids`` (matches
    the HF ``"EOT is the highest token id"`` convention).

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        vocab_size: Integer, text vocabulary size.
        hidden_size: Integer, model dimension.
        max_position_embeddings: Integer, maximum sequence length.
        num_hidden_layers: Integer, number of encoder layers.
        num_heads: Integer, attention heads per layer.
        intermediate_size: Integer, MLP expansion dimension per layer.
        layer_norm_eps: Float, layer normalization epsilon. Defaults
            to ``1e-5``.
        hidden_act: String, MLP activation. Defaults to ``"quick_gelu"``.
        **kwargs: Additional keyword arguments passed to ``Layer``.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_hidden_layers,
        num_heads,
        intermediate_size,
        layer_norm_eps=1e-5,
        hidden_act="quick_gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

        self.embeddings = OwlViTTextEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            name="embeddings",
        )
        self.encoder = OwlViTEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
            name="encoder",
        )
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps,
            name="final_layer_norm",
        )

    def call(self, input_ids, pool_indices=None):
        x = self.embeddings(input_ids)
        seq_len = ops.shape(x)[1]

        i = ops.arange(seq_len)[:, None]
        j = ops.arange(seq_len)[None, :]
        causal = ops.where(j > i, ops.cast(-1e9, x.dtype), ops.cast(0.0, x.dtype))
        causal = ops.reshape(causal, (1, 1, seq_len, seq_len))

        x = self.encoder(x, attention_mask=causal)
        x = self.final_layer_norm(x)

        if pool_indices is None:
            pool_indices = ops.cast(ops.argmax(input_ids, axis=-1), "int32")

        gather = ops.expand_dims(ops.expand_dims(pool_indices, -1), -1)
        pooled = ops.take_along_axis(x, gather, axis=1)
        pooled = ops.squeeze(pooled, axis=1)
        return x, pooled

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "max_position_embeddings": self.max_position_embeddings,
                "num_hidden_layers": self.num_hidden_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "layer_norm_eps": self.layer_norm_eps,
                "hidden_act": self.hidden_act,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTBoxPredictionHead(layers.Layer):
    """Box prediction MLP (``dense0`` → GELU → ``dense1`` → GELU → ``dense2``).

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        hidden_size: Integer, hidden size of the MLP.
        out_dim: Integer, output dimension. Defaults to ``4`` for
            ``(cx, cy, w, h)``.
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Input Shape:
        3D tensor: ``(batch_size, num_patches, hidden_size)``.

    Output Shape:
        3D tensor: ``(batch_size, num_patches, out_dim)``.
    """

    def __init__(self, hidden_size, out_dim=4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.dense0 = layers.Dense(hidden_size, name="dense0")
        self.dense1 = layers.Dense(hidden_size, name="dense1")
        self.dense2 = layers.Dense(out_dim, name="dense2")

    def call(self, image_features):
        x = self.dense0(image_features)
        x = ops.gelu(x, approximate=False)
        x = self.dense1(x)
        x = ops.gelu(x, approximate=False)
        return self.dense2(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "out_dim": self.out_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTClassPredictionHead(layers.Layer):
    """Text-conditional class predictor for OWL-ViT.

    Projects per-patch image features to the text dimension with
    ``dense0``, takes the L2-normalized cosine similarity against
    each text query, and applies a per-patch learned shift/scale
    via ``logit_shift`` / ``logit_scale``.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        query_dim: Integer, hidden size of the input image features.
        out_dim: Integer, hidden size of the text features the
            similarity is computed against.
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Returns:
        Tuple ``(pred_logits, image_class_embeds)`` where
        ``pred_logits`` has shape ``(B, num_patches, num_queries)`` and
        ``image_class_embeds`` has shape ``(B, num_patches, out_dim)``.
    """

    def __init__(self, query_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.query_dim = query_dim
        self.out_dim = out_dim

        self.dense0 = layers.Dense(out_dim, name="dense0")
        self.logit_shift = layers.Dense(1, name="logit_shift")
        self.logit_scale = layers.Dense(1, name="logit_scale")

    def call(self, image_embeds, query_embeds=None, query_mask=None):
        image_class_embeds = self.dense0(image_embeds)
        if query_embeds is None:
            b = ops.shape(image_class_embeds)[0]
            n = ops.shape(image_class_embeds)[1]
            pred_logits = ops.zeros((b, n, self.query_dim), dtype=image_embeds.dtype)
            return pred_logits, image_class_embeds

        image_norm = (
            ops.sqrt(
                ops.sum(image_class_embeds * image_class_embeds, axis=-1, keepdims=True)
                + 1e-12
            )
            + 1e-6
        )
        image_class_embeds_n = image_class_embeds / image_norm

        query_norm = (
            ops.sqrt(
                ops.sum(query_embeds * query_embeds, axis=-1, keepdims=True) + 1e-12
            )
            + 1e-6
        )
        query_embeds_n = query_embeds / query_norm

        pred_logits = ops.matmul(
            image_class_embeds_n, ops.transpose(query_embeds_n, (0, 2, 1))
        )

        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = ops.elu(logit_scale) + 1.0

        pred_logits = (pred_logits + logit_shift) * logit_scale

        if query_mask is not None:
            mask = ops.expand_dims(ops.cast(query_mask, "bool"), axis=-2)
            very_neg = ops.cast(ops.full_like(pred_logits, -1e30), pred_logits.dtype)
            pred_logits = ops.where(mask, pred_logits, very_neg)

        return pred_logits, image_class_embeds

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "query_dim": self.query_dim,
                "out_dim": self.out_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class OwlViTSplitBatchQueries(layers.Layer):
    """Split a flat ``(B*Q, ...)`` tensor into a per-image ``(B, Q, ...)`` tensor.

    OwlViT receives text queries flattened across the batch
    (``B * Q`` rows). This layer reshapes that flat tensor back into
    ``(B, Q, ...)`` using a separate batch-reference input to recover
    ``B`` at runtime, so it works correctly under the Keras Functional
    API where the leading ``B*Q`` dim is unknown at trace time.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Input Shape:
        Two tensors:
        - ``flat``: ``(B*Q, ..., last_dim)``
        - ``batch_ref``: any tensor whose first dim is ``B``.

    Output Shape:
        ``(B, Q, ..., last_dim)``.
    """

    def call(self, flat, batch_ref):
        b = ops.shape(batch_ref)[0]
        last = flat.shape[-1]
        return ops.reshape(flat, (b, -1, last))


def compute_box_bias(num_patches_height, num_patches_width):
    """Constant log-space box bias added to raw ``box_head`` outputs.

    Each patch's default predicted box is biased toward its grid
    location with a one-patch size, mirroring HF's ``compute_box_bias``.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        num_patches_height: Integer, vertical patch count.
        num_patches_width: Integer, horizontal patch count.

    Returns:
        Tensor of shape ``(num_patches_height * num_patches_width, 4)``
        containing the log-odds bias appended to the box head output.
    """
    nh = num_patches_height
    nw = num_patches_width
    x = ops.arange(1, nw + 1, dtype="float32")
    y = ops.arange(1, nh + 1, dtype="float32")
    xx, yy = ops.meshgrid(x, y, indexing="xy")
    box_coords = ops.stack([xx / float(nw), yy / float(nh)], axis=-1)
    box_coords = ops.reshape(box_coords, (nh * nw, 2))
    box_coords = ops.clip(box_coords, 0.0, 1.0)

    box_coord_bias = ops.log(box_coords + 1e-4) - ops.log1p(-box_coords + 1e-4)

    box_size = ops.stack(
        [
            ops.ones((nh * nw,), dtype="float32") / float(nw),
            ops.ones((nh * nw,), dtype="float32") / float(nh),
        ],
        axis=-1,
    )
    box_size_bias = ops.log(box_size + 1e-4) - ops.log1p(-box_size + 1e-4)

    return ops.concatenate([box_coord_bias, box_size_bias], axis=-1)
