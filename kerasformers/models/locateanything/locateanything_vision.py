import keras
from keras import layers, ops


def moonvit_2d_angles(height, width, head_dim, theta_base=10000.0):
    """Per-patch 2D-RoPE angles for a height x width grid, row-major flattened.

    Mirrors MoonViT's complex ``Rope2DPosEmb`` as real angles: for pair ``2m`` the
    angle is ``col * f[m]`` (x axis), for pair ``2m+1`` it is ``row * f[m]`` (y
    axis), with ``f[m] = theta_base**(-4m/head_dim)``. Returns ``(L, head_dim/2)``.
    """
    quarter = head_dim // 4
    m = ops.arange(0, head_dim, 4, dtype="float32")[:quarter] / head_dim
    freqs = 1.0 / ops.power(theta_base, m)  # (head_dim/4,)
    rows = ops.cast(ops.repeat(ops.arange(height), width), "float32")  # y, (L,)
    cols = ops.cast(ops.tile(ops.arange(width), (height,)), "float32")  # x, (L,)
    x_ang = cols[:, None] * freqs[None, :]  # (L, head_dim/4)
    y_ang = rows[:, None] * freqs[None, :]  # (L, head_dim/4)
    angles = ops.reshape(ops.stack([x_ang, y_ang], axis=-1), (height * width, -1))
    return angles  # (L, head_dim/2): [x0,y0,x1,y1,...]


def apply_rope_2d(x, cos, sin):
    """Interleaved-pair rotation. ``x`` (L, heads, head_dim); ``cos``/``sin``
    (L, head_dim/2). Rotates each adjacent pair (x[2k], x[2k+1]) by angle k."""
    leng, heads, head_dim = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    x = ops.reshape(x, (leng, heads, head_dim // 2, 2))
    xe, xo = x[..., 0], x[..., 1]
    c = cos[:, None, :]
    s = sin[:, None, :]
    oe = xe * c - xo * s
    oo = xe * s + xo * c
    return ops.reshape(ops.stack([oe, oo], axis=-1), (leng, heads, head_dim))


@keras.saving.register_keras_serializable(package="kerasformers")
class MoonVitMLP(layers.Layer):
    """Two-layer MLP with tanh-approximate GELU (MoonViT ``MLP2``)."""

    def __init__(self, hidden_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.fc0 = layers.Dense(mlp_dim, use_bias=True, name="fc0")
        self.fc1 = layers.Dense(hidden_dim, use_bias=True, name="fc1")

    def call(self, x):
        return self.fc1(ops.gelu(self.fc0(x), approximate=True))

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class MoonVitEncoderLayer(layers.Layer):
    """Pre-norm MoonViT block: full self-attention (with 2D rope) + MLP."""

    def __init__(self, hidden_dim, num_heads, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.head_dim = hidden_dim // num_heads
        self.norm0 = layers.LayerNormalization(epsilon=1e-5, name="norm0")
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.wqkv = layers.Dense(hidden_dim * 3, use_bias=True, name="wqkv")
        self.wo = layers.Dense(hidden_dim, use_bias=True, name="wo")
        self.mlp = MoonVitMLP(hidden_dim, mlp_dim, name="mlp")

    def call(self, x, cos, sin):
        leng = int(x.shape[0])
        residual = x
        h = self.norm0(x)
        qkv = ops.reshape(self.wqkv(h), (leng, 3, self.num_heads, self.head_dim))
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (L, heads, head_dim)
        q = apply_rope_2d(q, cos, sin)
        k = apply_rope_2d(k, cos, sin)
        q = ops.transpose(q, (1, 0, 2))  # (heads, L, head_dim)
        k = ops.transpose(k, (1, 0, 2))
        v = ops.transpose(v, (1, 0, 2))
        scale = self.head_dim**-0.5
        attn = ops.softmax(ops.matmul(q, ops.transpose(k, (0, 2, 1))) * scale, axis=-1)
        out = ops.matmul(attn, v)  # (heads, L, head_dim)
        out = ops.reshape(ops.transpose(out, (1, 0, 2)), (leng, self.hidden_dim))
        x = residual + self.wo(out)
        return x + self.mlp(self.norm1(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class LocateAnythingVisionModel(layers.Layer):
    """MoonViT-SO-400M: native-resolution packed ViT.

    Conv2d patch embedding + bicubic-interpolated learnable 2D position
    embedding, 27 pre-norm blocks with 2D rotary attention (each image attended
    independently — equivalent to the source's block-diagonal packed attention),
    a final LayerNorm, then a ``merge_kernel`` (2x2) patch merge that
    concatenates neighborhoods into ``hidden*4`` tokens for the projector.
    """

    def __init__(
        self,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_dim=4304,
        patch_size=14,
        init_pos_h=64,
        init_pos_w=64,
        merge_kernel=(2, 2),
        in_channels=3,
        rope_theta=10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.patch_size = patch_size
        self.init_pos_h = init_pos_h
        self.init_pos_w = init_pos_w
        self.merge_kernel = tuple(merge_kernel)
        self.in_channels = in_channels
        self.rope_theta = rope_theta
        self.head_dim = embed_dim // num_heads

        self.patch_proj = layers.Conv2D(
            embed_dim, patch_size, strides=patch_size, use_bias=True, name="patch_proj"
        )
        self.blocks = [
            MoonVitEncoderLayer(embed_dim, num_heads, mlp_dim, name=f"block_{i}")
            for i in range(depth)
        ]
        self.final_norm = layers.LayerNormalization(epsilon=1e-5, name="final_norm")

    def build(self, _):
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(self.init_pos_h, self.init_pos_w, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.patch_proj.build(
            (None, self.patch_size, self.patch_size, self.in_channels)
        )
        self.built = True

    def interp_pos_emb(self, height, width):
        pe = ops.convert_to_tensor(
            self.pos_emb
        )  # materialize (jax resize rejects a Variable)
        if height == self.init_pos_h and width == self.init_pos_w:
            return ops.reshape(pe, (height * width, self.embed_dim))
        resized = ops.image.resize(pe, (height, width), interpolation="bicubic")
        return ops.reshape(resized, (height * width, self.embed_dim))

    def embed_patches(self, patches):
        # patches: (L, C, P, P) or (L, P, P, C) -> (L, embed_dim)
        if int(patches.shape[1]) == self.in_channels:
            patches = ops.transpose(patches, (0, 2, 3, 1))
        x = self.patch_proj(ops.cast(patches, self.compute_dtype))
        return ops.reshape(x, (int(patches.shape[0]), self.embed_dim))

    def merge(self, x, height, width):
        kh, kw = self.merge_kernel
        nh, nw = height // kh, width // kw
        x = ops.reshape(x, (nh, kh, nw, kw, self.embed_dim))
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        return ops.reshape(x, (nh * nw, kh * kw * self.embed_dim))

    def call(self, pixel_values, grid_hws):
        grids = [(int(h), int(w)) for h, w in grid_hws]
        outputs = []
        start = 0
        for height, width in grids:
            leng = height * width
            patches = pixel_values[start : start + leng]
            start += leng
            x = self.embed_patches(patches) + ops.cast(
                self.interp_pos_emb(height, width), self.compute_dtype
            )
            angles = moonvit_2d_angles(height, width, self.head_dim, self.rope_theta)
            cos = ops.cast(ops.cos(angles), self.compute_dtype)
            sin = ops.cast(ops.sin(angles), self.compute_dtype)
            for block in self.blocks:
                x = block(x, cos, sin)
            x = self.final_norm(x)
            outputs.append(self.merge(x, height, width))
        return ops.concatenate(outputs, axis=0)  # (total_merged, embed_dim*merge)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "patch_size": self.patch_size,
                "init_pos_h": self.init_pos_h,
                "init_pos_w": self.init_pos_w,
                "merge_kernel": self.merge_kernel,
                "in_channels": self.in_channels,
                "rope_theta": self.rope_theta,
            }
        )
        return config
