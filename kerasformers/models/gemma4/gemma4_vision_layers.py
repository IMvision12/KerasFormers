import keras
from keras import layers, ops


def gelu_pytorch_tanh(x):
    return keras.activations.gelu(x, approximate=True)


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return ops.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=2):
    cos = ops.expand_dims(cos, unsqueeze_dim)
    sin = ops.expand_dims(sin, unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_multidimensional_rope(x, cos, sin, ndim=2, unsqueeze_dim=2):
    # head_dim is split into ndim equal parts; each part is rotated with the
    # cos/sin for its spatial dimension, then concatenated back.
    channels = ops.shape(x)[-1]
    per = 2 * (channels // (2 * ndim))
    out = []
    for k in range(ndim):
        sl = slice(k * per, (k + 1) * per)
        out.append(
            apply_rotary_pos_emb(x[..., sl], cos[..., sl], sin[..., sl], unsqueeze_dim)
        )
    return ops.concatenate(out, axis=-1)


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4RMSNorm(layers.Layer):
    """RMSNorm computed in float32, with an optional learnable scale."""

    def __init__(self, eps=1e-6, with_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.with_scale = with_scale

    def build(self, input_shape):
        if self.with_scale:
            self.weight = self.add_weight(
                shape=(input_shape[-1],),
                initializer="ones",
                trainable=True,
                name="weight",
            )

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps
        x = x * ops.power(variance, -0.5)
        if self.with_scale:
            x = x * ops.cast(self.weight, "float32")
        return ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps, "with_scale": self.with_scale})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class ClippableDense(layers.Layer):
    """A bias-free Dense with optional input/output clamping.

    Gemma 4 stores per-projection clip bounds (default +-inf, a no-op). When a
    checkpoint provides finite bounds the activations are clamped before and
    after the matmul.
    """

    def __init__(self, units, use_clipped_linears=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_clipped_linears = use_clipped_linears

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            name="kernel",
        )
        if self.use_clipped_linears:
            inf = float("inf")
            self.input_min = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(-inf),
                trainable=False,
                name="input_min",
            )
            self.input_max = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(inf),
                trainable=False,
                name="input_max",
            )
            self.output_min = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(-inf),
                trainable=False,
                name="output_min",
            )
            self.output_max = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(inf),
                trainable=False,
                name="output_max",
            )

    def call(self, x):
        if self.use_clipped_linears:
            x = ops.clip(
                x, ops.cast(self.input_min, x.dtype), ops.cast(self.input_max, x.dtype)
            )
        x = ops.matmul(x, ops.cast(self.kernel, x.dtype))
        if self.use_clipped_linears:
            x = ops.clip(
                x,
                ops.cast(self.output_min, x.dtype),
                ops.cast(self.output_max, x.dtype),
            )
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {"units": self.units, "use_clipped_linears": self.use_clipped_linears}
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionRotaryEmbedding(layers.Layer):
    """2-D rotary embedding: computes cos/sin per spatial dim from (x, y) patch ids."""

    def __init__(self, head_dim, rope_theta=100.0, ndim=2, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.ndim = ndim
        spatial_dim = head_dim // 2
        idx = ops.arange(0, spatial_dim, 2, dtype="float32")
        self.inv_freq = 1.0 / (rope_theta ** (idx / spatial_dim))

    def call(self, position_ids):
        # position_ids: (batch, num_patches, ndim)
        inv = ops.cast(self.inv_freq, "float32")[None, :, None]
        inv = ops.broadcast_to(inv, (ops.shape(position_ids)[0], ops.shape(inv)[1], 1))
        all_cos, all_sin = [], []
        for i in range(self.ndim):
            pos = ops.cast(position_ids[:, :, i], "float32")[:, None, :]
            freqs = ops.transpose(ops.matmul(inv, pos), (0, 2, 1))
            emb = ops.concatenate([freqs, freqs], axis=-1)
            all_cos.append(ops.cos(emb))
            all_sin.append(ops.sin(emb))
        cos = ops.concatenate(all_cos, axis=-1)
        sin = ops.concatenate(all_sin, axis=-1)
        return cos, sin

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
                "rope_theta": self.rope_theta,
                "ndim": self.ndim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionPatchEmbedder(layers.Layer):
    """Projects flattened patches and adds a 2-D (x, y) learned position embedding."""

    def __init__(self, hidden_size, patch_size, position_embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.position_embedding_size = position_embedding_size
        self.input_proj = layers.Dense(hidden_size, use_bias=False, name="input_proj")

    def build(self, input_shape):
        self.position_embedding_table = self.add_weight(
            shape=(2, self.position_embedding_size, self.hidden_size),
            initializer="ones",
            name="position_embedding_table",
        )

    def call(self, pixel_values, pixel_position_ids, padding_positions):
        pixel_values = 2.0 * (pixel_values - 0.5)
        hidden = self.input_proj(ops.cast(pixel_values, self.input_proj.compute_dtype))
        clamped = ops.maximum(pixel_position_ids, 0)
        x_emb = ops.take(self.position_embedding_table[0], clamped[..., 0], axis=0)
        y_emb = ops.take(self.position_embedding_table[1], clamped[..., 1], axis=0)
        pos = x_emb + y_emb
        pos = ops.where(ops.expand_dims(padding_positions, -1), 0.0, pos)
        return hidden + ops.cast(pos, hidden.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "patch_size": self.patch_size,
                "position_embedding_size": self.position_embedding_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionMLP(layers.Layer):
    """Gated SwiGLU MLP with clippable linears."""

    def __init__(self, intermediate_size, use_clipped_linears=True, **kwargs):
        super().__init__(**kwargs)
        self.intermediate_size = intermediate_size
        self.gate_proj = ClippableDense(
            intermediate_size, use_clipped_linears, name="gate_proj"
        )
        self.up_proj = ClippableDense(
            intermediate_size, use_clipped_linears, name="up_proj"
        )

    def build(self, input_shape):
        self.down_proj = ClippableDense(
            input_shape[-1], self.gate_proj.use_clipped_linears, name="down_proj"
        )

    def call(self, x):
        return self.down_proj(gelu_pytorch_tanh(self.gate_proj(x)) * self.up_proj(x))

    def get_config(self):
        config = super().get_config()
        config.update({"intermediate_size": self.intermediate_size})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionAttention(layers.Layer):
    """Bidirectional MHA with q/k/v RMSNorm and 2-D rotary embedding."""

    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_dim,
        eps=1e-6,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = ClippableDense(
            num_heads * head_dim, use_clipped_linears, name="q_proj"
        )
        self.k_proj = ClippableDense(
            num_kv_heads * head_dim, use_clipped_linears, name="k_proj"
        )
        self.v_proj = ClippableDense(
            num_kv_heads * head_dim, use_clipped_linears, name="v_proj"
        )
        self.o_proj = ClippableDense(
            num_heads * head_dim, use_clipped_linears, name="o_proj"
        )
        self.q_norm = Gemma4RMSNorm(eps=eps, name="q_norm")
        self.k_norm = Gemma4RMSNorm(eps=eps, name="k_norm")
        self.v_norm = Gemma4RMSNorm(eps=eps, with_scale=False, name="v_norm")

    def call(self, hidden_states, cos, sin, attention_mask=None):
        b = ops.shape(hidden_states)[0]
        n = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.q_proj(hidden_states), (b, n, self.num_heads, self.head_dim)
        )
        q = apply_multidimensional_rope(self.q_norm(q), cos, sin)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.reshape(
            self.k_proj(hidden_states), (b, n, self.num_kv_heads, self.head_dim)
        )
        k = apply_multidimensional_rope(self.k_norm(k), cos, sin)
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.reshape(
            self.v_proj(hidden_states), (b, n, self.num_kv_heads, self.head_dim)
        )
        v = ops.transpose(self.v_norm(v), (0, 2, 1, 3))

        # scaling is 1.0 in Gemma4 vision (baked into the q/k norms)
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, v)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, n, self.num_heads * self.head_dim)
        )
        return self.o_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionEncoderLayer(layers.Layer):
    """Sandwich-norm transformer block (4 RMSNorms around attention and MLP)."""

    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        eps=1e-6,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_attn = Gemma4VisionAttention(
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            use_clipped_linears,
            name="self_attn",
        )
        self.mlp = Gemma4VisionMLP(intermediate_size, use_clipped_linears, name="mlp")
        self.input_layernorm = Gemma4RMSNorm(eps=eps, name="input_layernorm")
        self.post_attention_layernorm = Gemma4RMSNorm(
            eps=eps, name="post_attention_layernorm"
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            eps=eps, name="pre_feedforward_layernorm"
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            eps=eps, name="post_feedforward_layernorm"
        )

    def call(self, hidden_states, cos, sin, attention_mask=None):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, cos, sin, attention_mask)
        h = self.post_attention_layernorm(h)
        hidden_states = residual + h

        residual = hidden_states
        h = self.pre_feedforward_layernorm(hidden_states)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        return residual + h


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionPooler(layers.Layer):
    """2-D spatial average pooling of patches, then sqrt(hidden) scaling in float32."""

    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.root_hidden_size = hidden_size**0.5

    def call(self, hidden_states, pixel_position_ids, padding_positions, output_length):
        hidden_states = ops.where(
            ops.expand_dims(padding_positions, -1), 0.0, hidden_states
        )
        seq = ops.shape(hidden_states)[1]
        if int(seq) != int(output_length):
            hidden_states, _ = self.avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        return ops.cast(hidden_states, "float32") * self.root_hidden_size

    def avg_pool_by_positions(self, hidden_states, pixel_position_ids, length):
        seq = int(ops.shape(hidden_states)[1])
        k = int((seq // int(length)) ** 0.5)
        clamped = ops.maximum(pixel_position_ids, 0)
        max_x = ops.max(clamped[..., 0], axis=-1, keepdims=True) + 1
        kernel = ops.floor_divide(clamped, k)
        kernel = kernel[..., 0] + ops.floor_divide(max_x, k) * kernel[..., 1]
        weights = ops.cast(
            ops.one_hot(ops.cast(kernel, "int32"), int(length)), "float32"
        ) / (k * k)
        out = ops.matmul(
            ops.transpose(weights, (0, 2, 1)), ops.cast(hidden_states, "float32")
        )
        mask = ops.logical_not(ops.all(weights == 0, axis=1))
        return ops.cast(out, hidden_states.dtype), mask

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4MultimodalEmbedder(layers.Layer):
    """Projects vision/audio soft tokens into the text embedding space."""

    def __init__(self, text_hidden_size, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.text_hidden_size = text_hidden_size
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            eps=eps, with_scale=False, name="embedding_pre_projection_norm"
        )
        self.embedding_projection = layers.Dense(
            text_hidden_size, use_bias=False, name="embedding_projection"
        )

    def call(self, inputs_embeds):
        return self.embedding_projection(
            self.embedding_pre_projection_norm(inputs_embeds)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"text_hidden_size": self.text_hidden_size})
        return config
