import math

import keras
from keras import layers, ops

from .gemma4_vision_layers import ClippableDense, Gemma4RMSNorm


def glu(x, axis=-1):
    a, b = ops.split(x, 2, axis=axis)
    return a * ops.sigmoid(b)


def block_indices(num_blocks, context_size, chunk_size):
    starts = ops.arange(num_blocks) * chunk_size
    offsets = ops.arange(context_size)
    return starts[:, None] + offsets[None, :]


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioRelPositionalEncoding(layers.Layer):
    """Sinusoidal relative position table of shape (1, context // 2 + 1, hidden)."""

    def __init__(self, hidden_size, context_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.context_size = context_size
        num_timescales = hidden_size // 2
        log_increment = math.log(10000.0) / max(num_timescales - 1, 1)
        self.inv_timescales = ops.exp(
            ops.arange(num_timescales, dtype="float32") * -log_increment
        )

    def compute(self, dtype="float32"):
        position_ids = ops.cast(ops.arange(self.context_size // 2, -1, -1), "float32")[
            :, None
        ]
        scaled = position_ids * self.inv_timescales[None, :]
        emb = ops.concatenate([ops.sin(scaled), ops.cos(scaled)], axis=-1)
        return ops.cast(emb[None], dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_size": self.hidden_size, "context_size": self.context_size}
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioAttention(layers.Layer):
    """Chunked local attention with relative-position bias and tanh logit cap."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        chunk_size,
        context_left,
        context_right,
        logit_cap=50.0,
        invalid_logits=-1e9,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.max_past_horizon = context_left - 1
        self.max_future_horizon = context_right
        self.context_size = chunk_size + self.max_past_horizon + self.max_future_horizon
        self.logit_cap = logit_cap
        self.invalid_logits = invalid_logits
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)

        units = num_heads * self.head_dim
        self.q_proj = ClippableDense(units, use_clipped_linears, name="q_proj")
        self.k_proj = ClippableDense(units, use_clipped_linears, name="k_proj")
        self.v_proj = ClippableDense(units, use_clipped_linears, name="v_proj")
        self.post = ClippableDense(hidden_size, use_clipped_linears, name="post")
        self.relative_k_proj = layers.Dense(
            units, use_bias=False, name="relative_k_proj"
        )

    def build(self, input_shape):
        self.per_dim_scale = self.add_weight(
            shape=(self.head_dim,), initializer="zeros", name="per_dim_scale"
        )

    def to_block(self, x, seq_len, num_blocks):
        pad = num_blocks * self.chunk_size - seq_len
        x = ops.pad(x, [[0, 0], [0, pad], [0, 0], [0, 0]])
        b = ops.shape(x)[0]
        return ops.reshape(
            x, (b, num_blocks, self.chunk_size, self.num_heads, self.head_dim)
        )

    def extract_context(self, x, num_blocks):
        x = ops.pad(
            x,
            [
                [0, 0],
                [self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1],
                [0, 0],
                [0, 0],
            ],
        )
        idx = block_indices(num_blocks, self.context_size, self.chunk_size)
        # take with a [num_blocks, context] index over the seq axis yields
        # [B, num_blocks, context, H, D], matching HF's unfold + movedim.
        return ops.take(x, idx, axis=1)

    def rel_shift(self, x, num_blocks, position_length):
        b = ops.shape(x)[0]
        x = ops.pad(
            x,
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, self.context_size + 1 - position_length],
            ],
        )
        x = ops.reshape(
            x,
            (b, self.num_heads, num_blocks, self.chunk_size * (self.context_size + 1)),
        )
        x = x[..., : self.chunk_size * self.context_size]
        return ops.reshape(
            x, (b, self.num_heads, num_blocks, self.chunk_size, self.context_size)
        )

    def call(self, hidden_states, position_embeddings, attention_mask=None):
        b = ops.shape(hidden_states)[0]
        seq_len = int(ops.shape(hidden_states)[1])
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        shape = (b, seq_len, self.num_heads, self.head_dim)

        q = ops.cast(ops.reshape(self.q_proj(hidden_states), shape), "float32")
        k = ops.cast(ops.reshape(self.k_proj(hidden_states), shape), "float32")
        v = ops.cast(ops.reshape(self.v_proj(hidden_states), shape), "float32")
        q = q * self.q_scale * ops.softplus(ops.cast(self.per_dim_scale, "float32"))
        k = k * self.k_scale

        q = self.to_block(q, seq_len, num_blocks)
        k = self.extract_context(k, num_blocks)
        v = self.extract_context(v, num_blocks)

        rel_k = self.relative_k_proj(position_embeddings)
        rel_k = ops.cast(
            ops.reshape(rel_k, (-1, self.num_heads, self.head_dim)), "float32"
        )
        position_length = int(ops.shape(rel_k)[0])

        queries = ops.transpose(q, (0, 3, 1, 2, 4))
        matrix_ac = ops.matmul(queries, ops.transpose(k, (0, 3, 1, 4, 2)))

        queries_flat = ops.reshape(queries, (b, self.num_heads, -1, self.head_dim))
        matrix_bd = ops.matmul(queries_flat, ops.transpose(rel_k, (1, 2, 0)))
        matrix_bd = ops.reshape(
            matrix_bd, (b, self.num_heads, num_blocks, self.chunk_size, position_length)
        )
        matrix_bd = self.rel_shift(matrix_bd, num_blocks, position_length)

        attn = (matrix_ac + matrix_bd) / self.logit_cap
        attn = ops.tanh(attn) * self.logit_cap
        if attention_mask is not None:
            attn = ops.where(attention_mask, attn, self.invalid_logits)
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, ops.transpose(v, (0, 3, 1, 2, 4)))
        out = ops.transpose(out, (0, 2, 3, 1, 4))
        out = ops.reshape(out, (b, num_blocks * self.chunk_size, self.hidden_size))
        out = out[:, :seq_len]
        return self.post(ops.cast(out, hidden_states.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "chunk_size": self.chunk_size,
                "context_left": self.max_past_horizon + 1,
                "context_right": self.max_future_horizon,
                "logit_cap": self.logit_cap,
                "invalid_logits": self.invalid_logits,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioSubSampleConvLayer(layers.Layer):
    """3x3 stride-2 conv (pad 1) with channel LayerNorm and ReLU; halves time/freq."""

    def __init__(self, out_channels, norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.norm_eps = norm_eps
        self.pad = layers.ZeroPadding2D(padding=1)
        self.conv = layers.Conv2D(
            out_channels, 3, strides=2, padding="valid", use_bias=False, name="conv"
        )
        self.norm = layers.LayerNormalization(
            epsilon=norm_eps, center=False, scale=True, name="norm"
        )

    def call(self, x, mask=None):
        if mask is not None:
            x = x * ops.cast(mask[:, :, None, None], x.dtype)
        x = self.conv(self.pad(x))
        x = keras.activations.relu(self.norm(x))
        if mask is not None:
            mask = mask[:, ::2]
        return x, mask

    def get_config(self):
        config = super().get_config()
        config.update({"out_channels": self.out_channels, "norm_eps": self.norm_eps})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioSubSampleConvProjection(layers.Layer):
    """Two stride-2 conv layers then a linear projection to hidden_size."""

    def __init__(self, conv_channels, hidden_size, norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.conv_channels = list(conv_channels)
        self.hidden_size = hidden_size
        self.norm_eps = norm_eps
        self.layer0 = Gemma4AudioSubSampleConvLayer(
            conv_channels[0], norm_eps, name="layer0"
        )
        self.layer1 = Gemma4AudioSubSampleConvLayer(
            conv_channels[1], norm_eps, name="layer1"
        )
        self.input_proj_linear = layers.Dense(
            hidden_size, use_bias=False, name="input_proj_linear"
        )

    def call(self, input_features, mask=None):
        x = input_features[..., None]
        x, mask = self.layer0(x, mask)
        x, mask = self.layer1(x, mask)
        b = ops.shape(x)[0]
        seq = int(ops.shape(x)[1])
        x = ops.reshape(x, (b, seq, -1))
        return self.input_proj_linear(x), mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "conv_channels": self.conv_channels,
                "hidden_size": self.hidden_size,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioFeedForward(layers.Layer):
    """Half-step conformer feed-forward (pre/post RMSNorm, SiLU, residual weight)."""

    def __init__(
        self,
        hidden_size,
        norm_eps=1e-6,
        residual_weight=0.5,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.residual_weight = residual_weight
        self.ffw_layer_1 = ClippableDense(
            hidden_size * 4, use_clipped_linears, name="ffw_layer_1"
        )
        self.ffw_layer_2 = ClippableDense(
            hidden_size, use_clipped_linears, name="ffw_layer_2"
        )
        self.pre_layer_norm = Gemma4RMSNorm(eps=norm_eps, name="pre_layer_norm")
        self.post_layer_norm = Gemma4RMSNorm(eps=norm_eps, name="post_layer_norm")

    def call(self, hidden_states):
        residual = hidden_states
        h = self.pre_layer_norm(hidden_states)
        h = self.ffw_layer_1(h)
        h = keras.activations.silu(h)
        h = self.ffw_layer_2(h)
        h = self.post_layer_norm(h)
        return h * self.residual_weight + residual

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_size": self.hidden_size, "residual_weight": self.residual_weight}
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioLightConv1d(layers.Layer):
    """GLU + causal depthwise conv1d conformer module."""

    def __init__(
        self,
        hidden_size,
        conv_kernel_size=5,
        norm_eps=1e-6,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.conv_kernel_size = conv_kernel_size
        self.left_pad = conv_kernel_size - 1
        self.linear_start = ClippableDense(
            hidden_size * 2, use_clipped_linears, name="linear_start"
        )
        self.linear_end = ClippableDense(
            hidden_size, use_clipped_linears, name="linear_end"
        )
        self.pre_layer_norm = Gemma4RMSNorm(eps=norm_eps, name="pre_layer_norm")
        self.conv_norm = Gemma4RMSNorm(eps=norm_eps, name="conv_norm")

    def build(self, input_shape):
        self.depthwise_kernel = self.add_weight(
            shape=(self.conv_kernel_size, self.hidden_size, 1),
            initializer="glorot_uniform",
            name="depthwise_conv1d_kernel",
        )

    def call(self, hidden_states):
        residual = hidden_states
        h = self.pre_layer_norm(hidden_states)
        h = self.linear_start(h)
        h = glu(h, axis=-1)
        h = ops.pad(h, [[0, 0], [self.left_pad, 0], [0, 0]])
        h = ops.depthwise_conv(
            h, ops.cast(self.depthwise_kernel, h.dtype), strides=1, padding="valid"
        )
        h = self.conv_norm(h)
        h = keras.activations.silu(h)
        h = self.linear_end(h)
        return h + residual

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_size": self.hidden_size, "conv_kernel_size": self.conv_kernel_size}
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioLayer(layers.Layer):
    """Conformer block: FF1, attention, light conv1d, FF2, sandwiched by RMSNorms."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        chunk_size,
        context_left,
        context_right,
        conv_kernel_size=5,
        norm_eps=1e-6,
        residual_weight=0.5,
        logit_cap=50.0,
        invalid_logits=-1e9,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feed_forward1 = Gemma4AudioFeedForward(
            hidden_size,
            norm_eps,
            residual_weight,
            use_clipped_linears,
            name="feed_forward1",
        )
        self.feed_forward2 = Gemma4AudioFeedForward(
            hidden_size,
            norm_eps,
            residual_weight,
            use_clipped_linears,
            name="feed_forward2",
        )
        self.self_attn = Gemma4AudioAttention(
            hidden_size,
            num_heads,
            chunk_size,
            context_left,
            context_right,
            logit_cap,
            invalid_logits,
            use_clipped_linears,
            name="self_attn",
        )
        self.lconv1d = Gemma4AudioLightConv1d(
            hidden_size, conv_kernel_size, norm_eps, use_clipped_linears, name="lconv1d"
        )
        self.norm_pre_attn = Gemma4RMSNorm(eps=norm_eps, name="norm_pre_attn")
        self.norm_post_attn = Gemma4RMSNorm(eps=norm_eps, name="norm_post_attn")
        self.norm_out = Gemma4RMSNorm(eps=norm_eps, name="norm_out")

    def call(self, hidden_states, position_embeddings, attention_mask=None):
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states
        h = self.norm_pre_attn(hidden_states)
        h = self.self_attn(h, position_embeddings, attention_mask)
        h = self.norm_post_attn(h)
        hidden_states = h + residual
        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)
        return self.norm_out(hidden_states)
