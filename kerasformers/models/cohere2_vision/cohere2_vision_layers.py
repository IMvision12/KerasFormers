import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2VisionAttention(layers.Layer):
    """SigLIP full (bidirectional) self-attention — all projections biased."""

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.query = layers.Dense(embed_dim, name="query")
        self.key = layers.Dense(embed_dim, name="key")
        self.value = layers.Dense(embed_dim, name="value")
        self.output_proj = layers.Dense(embed_dim, name="output_proj")

    def call(self, x):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]

        def split(t):
            return ops.transpose(
                ops.reshape(t, (b, s, self.num_heads, self.head_dim)), (0, 2, 1, 3)
            )

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (b, s, self.embed_dim))
        return self.output_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2VisionLayer(layers.Layer):
    """One SigLIP encoder block: pre-LN attention + pre-LN gelu-tanh MLP."""

    def __init__(self, embed_dim, mlp_dim, num_heads, norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=norm_eps, name="layer_norm1"
        )
        self.attention = Cohere2VisionAttention(embed_dim, num_heads, name="attention")
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=norm_eps, name="layer_norm2"
        )
        self.fc1 = layers.Dense(mlp_dim, name="fc1")
        self.fc2 = layers.Dense(embed_dim, name="fc2")

    def call(self, x):
        x = x + self.attention(self.layer_norm1(x))
        y = self.layer_norm2(x)
        y = self.fc2(ops.gelu(self.fc1(y), approximate=True))
        return x + y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2VisionTower(layers.Layer):
    """SigLIP vision encoder: conv patch embed + learned positions + blocks.

    Args:
        embed_dim / mlp_dim / num_layers / num_heads: Tower dims.
        image_size / patch_size: Patch geometry (512 / 16).
        norm_eps: LayerNorm epsilon.

    Call args:
        pixel_values: ``(num_tiles, H, W, 3)`` (or channels-first).

    Returns:
        ``(num_tiles, num_patches, embed_dim)``.
    """

    def __init__(
        self,
        embed_dim=1152,
        mlp_dim=4304,
        num_layers=27,
        num_heads=16,
        image_size=512,
        patch_size=16,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.norm_eps = norm_eps
        self.num_positions = (image_size // patch_size) ** 2
        self.patch_embed = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            data_format="channels_last",
            name="patch_embed",
        )
        self.position_embedding = layers.Embedding(
            self.num_positions, embed_dim, name="position_embedding"
        )
        self.blocks = [
            Cohere2VisionLayer(
                embed_dim, mlp_dim, num_heads, norm_eps, name=f"blocks_{i}"
            )
            for i in range(num_layers)
        ]
        self.post_layernorm = layers.LayerNormalization(
            epsilon=norm_eps, name="post_layernorm"
        )

    def call(self, pixel_values):
        if (
            pixel_values.shape[1] is not None
            and int(pixel_values.shape[1]) == 3
            and (pixel_values.shape[-1] is None or int(pixel_values.shape[-1]) != 3)
        ):
            pixel_values = ops.transpose(pixel_values, (0, 2, 3, 1))
        x = self.patch_embed(pixel_values)
        b = ops.shape(x)[0]
        x = ops.reshape(x, (b, -1, self.embed_dim))
        x = x + self.position_embedding(ops.arange(self.num_positions))[None]
        for block in self.blocks:
            x = block(x)
        return self.post_layernorm(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2VisionProjector(layers.Layer):
    """Pixel-shuffle + SwiGLU multimodal projector.

    Downsamples the square patch grid by ``downsample_factor`` (folding the
    factor into the channel dim), projects to ``alignment_intermediate_size``,
    applies a SwiGLU (``silu(gate) * x`` over the split halves), then projects
    to the text width.

    Args:
        vision_dim: Vision-tower hidden width.
        text_dim: Text-model hidden width.
        downsample_factor: Spatial downsample factor (2).
        intermediate_size: Projector hidden width (``alignment_intermediate_size``).
    """

    def __init__(
        self,
        vision_dim,
        text_dim,
        downsample_factor=2,
        intermediate_size=36864,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.downsample_factor = downsample_factor
        self.intermediate_size = intermediate_size
        self.linear_1 = layers.Dense(intermediate_size, name="linear_1")
        self.linear_2 = layers.Dense(text_dim, name="linear_2")

    def pixel_shuffle(self, x):
        b = ops.shape(x)[0]
        seq = int(x.shape[1])
        df = self.downsample_factor
        h = w = int(round(seq**0.5))
        x = ops.reshape(x, (b, w, h, -1))
        channels = int(x.shape[-1])
        x = ops.reshape(x, (b, w, h // df, channels * df))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, h // df, w // df, -1))
        x = ops.transpose(x, (0, 2, 1, 3))
        return x

    def call(self, image_features):
        x = self.pixel_shuffle(image_features)
        x = self.linear_1(x)
        gated, gate = ops.split(x, 2, axis=-1)
        x = ops.silu(gate) * gated
        x = self.linear_2(x)
        return ops.reshape(x, (-1, self.text_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_dim": self.vision_dim,
                "text_dim": self.text_dim,
                "downsample_factor": self.downsample_factor,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config
