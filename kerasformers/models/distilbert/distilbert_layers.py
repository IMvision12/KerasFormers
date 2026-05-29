import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kerasformers")
class DistilBertEmbeddings(layers.Layer):
    """Constructs DistilBERT's input embeddings.

    Sums learned word and absolute-position embeddings (no token-type / segment
    embeddings — DistilBERT has none), then applies LayerNorm and dropout.
    Position ids are derived with ``cumsum(ones_like) - 1`` so the layer stays
    shape-polymorphic across the TensorFlow / JAX / PyTorch backends.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Embedding / model dimension.
        max_position_embeddings: Size of the position-embedding table.
        layer_norm_eps: Epsilon for the embedding LayerNorm.
        dropout: Dropout rate applied to the summed embeddings.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        max_position_embeddings,
        layer_norm_eps=1e-12,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout_rate = dropout

        self.word_embeddings = layers.Embedding(
            vocab_size, embed_dim, name="word_embeddings"
        )
        self.position_embeddings = layers.Embedding(
            max_position_embeddings, embed_dim, name="position_embeddings"
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="LayerNorm"
        )
        self.dropout = layers.Dropout(dropout)

    def call(self, input_ids, training=None):
        position_ids = ops.cumsum(ops.ones_like(input_ids), axis=1) - 1
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(
            position_ids
        )
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "layer_norm_eps": self.layer_norm_eps,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class DistilBertSelfAttention(layers.Layer):
    """DistilBERT multi-head self-attention (the ``attention`` sub-block).

    Projects the input to query/key/value with ``q_lin`` / ``k_lin`` / ``v_lin``
    (all with bias), computes scaled dot-product attention with an additive
    padding mask, concatenates the heads, and applies the output projection
    ``out_lin`` (the residual + LayerNorm live in the encoder layer, matching
    Hugging Face's ``sa_layer_norm``).

    Args:
        embed_dim: Model dimension. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        attention_dropout: Dropout rate applied to the attention weights.
        block_prefix: Prefix for the projection names. Carries the encoder-layer
            index so each layer's weights get a unique path suffix (required for
            backbone weight-sharing across task heads).
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        attention_dropout=0.0,
        block_prefix=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.block_prefix = block_prefix if block_prefix is not None else "attention"
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        prefix = f"{self.block_prefix}_"
        self.q_lin = layers.Dense(embed_dim, name=prefix + "q_lin")
        self.k_lin = layers.Dense(embed_dim, name=prefix + "k_lin")
        self.v_lin = layers.Dense(embed_dim, name=prefix + "v_lin")
        self.out_lin = layers.Dense(embed_dim, name=prefix + "out_lin")
        self.dropout = layers.Dropout(attention_dropout)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.q_lin.build((None, input_dim))
        self.k_lin.build((None, input_dim))
        self.v_lin.build((None, input_dim))
        self.out_lin.build((None, self.embed_dim))
        self.built = True

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def transpose_for_scores(self, x):
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]
        x = ops.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states, attention_mask=None, training=None):
        query = self.transpose_for_scores(self.q_lin(hidden_states))
        key = self.transpose_for_scores(self.k_lin(hidden_states))
        value = self.transpose_for_scores(self.v_lin(hidden_states))

        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = ops.softmax(scores, axis=-1)
        probs = self.dropout(probs, training=training)

        context = ops.matmul(probs, value)
        context = ops.transpose(context, (0, 2, 1, 3))
        batch_size = ops.shape(context)[0]
        context = ops.reshape(context, (batch_size, -1, self.embed_dim))
        return self.out_lin(context)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class DistilBertFlattenChoices(layers.Layer):
    """Merge the multiple-choice axis into the batch: ``(B, C, S) -> (B*C, S)``."""

    def call(self, inputs):
        return ops.reshape(inputs, (-1, ops.shape(inputs)[-1]))

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])


@keras.saving.register_keras_serializable(package="kerasformers")
class DistilBertUnflattenChoices(layers.Layer):
    """Inverse of :class:`DistilBertFlattenChoices`: ``(B*C, 1) -> (B, C)``.

    Args:
        num_choices: Number of choices ``C`` to fold back out of the batch.
    """

    def __init__(self, num_choices, **kwargs):
        super().__init__(**kwargs)
        self.num_choices = num_choices

    def call(self, inputs):
        return ops.reshape(inputs, (-1, self.num_choices))

    def compute_output_shape(self, input_shape):
        return (None, self.num_choices)

    def get_config(self):
        config = super().get_config()
        config.update({"num_choices": self.num_choices})
        return config
