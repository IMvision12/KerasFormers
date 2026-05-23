"""Pure Keras 3 layers for Qwen3-VL (self-contained — no qwen2_vl imports).

Text: Qwen3 attention with per-head QK-norm and no qkv bias (RoPE applied after
the norm), SwiGLU MLP. Vision: LayerNorm blocks with a non-gated GELU MLP
(``gelu_pytorch_tanh``), Conv3d-as-Dense (biased) patch embed, and two patch
mergers (pre- vs post-shuffle LayerNorm) — the post-shuffle ones produce the
DeepStack features.
"""

import keras
from keras import layers, ops


def rotate_half(x):
    """Rotate the last dim by halves: ``[-x2, x1]`` (Llama/RoPE convention)."""
    half = x.shape[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary(t, cos, sin):
    """Standard rotary application ``t * cos + rotate_half(t) * sin``."""
    return (t * cos) + (rotate_half(t) * sin)


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLRMSNorm(layers.Layer):
    """RMSNorm: normalize by RMS in float32, then scale."""

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(input_shape[-1],), initializer="ones", trainable=True
        )
        self.built = True

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(variance + self.eps)
        return self.weight * ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLMLP(layers.Layer):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` (bias-free; Qwen3 text)."""

    def __init__(self, hidden_size, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = layers.Dense(
            intermediate_size, use_bias=False, name="gate_proj"
        )
        self.up_proj = layers.Dense(intermediate_size, use_bias=False, name="up_proj")
        self.down_proj = layers.Dense(hidden_size, use_bias=False, name="down_proj")

    def call(self, x):
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLTextAttention(layers.Layer):
    """Qwen3 GQA attention: no qkv bias, per-head RMSNorm on q and k (pre-RoPE)."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim**-0.5
        self.q_proj = layers.Dense(
            num_attention_heads * head_dim, use_bias=False, name="q_proj"
        )
        self.k_proj = layers.Dense(
            num_key_value_heads * head_dim, use_bias=False, name="k_proj"
        )
        self.v_proj = layers.Dense(
            num_key_value_heads * head_dim, use_bias=False, name="v_proj"
        )
        self.o_proj = layers.Dense(hidden_size, use_bias=False, name="o_proj")
        self.q_norm = Qwen3VLRMSNorm(eps=rms_norm_eps, name="q_norm")
        self.k_norm = Qwen3VLRMSNorm(eps=rms_norm_eps, name="k_norm")

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]
        q = self.q_norm(
            ops.reshape(
                self.q_proj(hidden_states),
                (b, q_len, self.num_attention_heads, self.head_dim),
            )
        )
        k = self.k_norm(
            ops.reshape(
                self.k_proj(hidden_states),
                (b, q_len, self.num_key_value_heads, self.head_dim),
            )
        )
        v = ops.reshape(
            self.v_proj(hidden_states),
            (b, q_len, self.num_key_value_heads, self.head_dim),
        )
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = ops.concatenate([past_k, k], axis=2)
            v = ops.concatenate([past_v, v], axis=2)
        new_kv = (k, v) if use_cache else None

        if self.num_key_value_groups > 1:
            k = ops.repeat(k, self.num_key_value_groups, axis=1)
            v = ops.repeat(v, self.num_key_value_groups, axis=1)

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, v)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)),
            (b, q_len, self.num_attention_heads * self.head_dim),
        )
        out = self.o_proj(out)
        return (out, new_kv) if use_cache else out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "rms_norm_eps": self.rms_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLTextDecoderLayer(layers.Layer):
    """Qwen3 decoder block: pre-norm QK-norm attention then pre-norm SwiGLU."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.input_layernorm = Qwen3VLRMSNorm(eps=rms_norm_eps, name="input_layernorm")
        self.self_attn = Qwen3VLTextAttention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            name="self_attn",
        )
        self.post_attention_layernorm = Qwen3VLRMSNorm(
            eps=rms_norm_eps, name="post_attention_layernorm"
        )
        self.mlp = Qwen3VLMLP(hidden_size, intermediate_size, name="mlp")

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states,
            cos,
            sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_kv) if use_cache else hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "rms_norm_eps": self.rms_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLVisionPatchEmbed(layers.Layer):
    """Per-patch linear projection with bias (HF's biased Conv3d as a Dense)."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.proj = layers.Dense(embed_dim, use_bias=True, name="proj")

    def call(self, x):
        return self.proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLVisionAttention(layers.Layer):
    """Full vision attention with 2D rotary positions + block-diagonal mask."""

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = layers.Dense(embed_dim * 3, use_bias=True, name="qkv")
        self.proj = layers.Dense(embed_dim, use_bias=True, name="proj")

    def call(self, hidden_states, cos, sin, attention_mask=None):
        seq = ops.shape(hidden_states)[0]
        qkv = ops.reshape(
            self.qkv(hidden_states), (seq, 3, self.num_heads, self.head_dim)
        )
        qkv = ops.transpose(qkv, (1, 0, 2, 3))
        query, key, value = qkv[0], qkv[1], qkv[2]

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)

        query = ops.expand_dims(ops.transpose(query, (1, 0, 2)), axis=0)
        key = ops.expand_dims(ops.transpose(key, (1, 0, 2)), axis=0)
        value = ops.expand_dims(ops.transpose(value, (1, 0, 2)), axis=0)

        attn = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, value)
        out = ops.transpose(out[0], (1, 0, 2))
        out = ops.reshape(out, (seq, self.embed_dim))
        return self.proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLVisionMLP(layers.Layer):
    """Non-gated vision MLP: ``fc2(gelu_tanh(fc1(x)))`` (biased)."""

    def __init__(self, hidden_size, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.linear_fc1 = layers.Dense(
            intermediate_size, use_bias=True, name="linear_fc1"
        )
        self.linear_fc2 = layers.Dense(hidden_size, use_bias=True, name="linear_fc2")

    def call(self, x):
        return self.linear_fc2(ops.gelu(self.linear_fc1(x), approximate=True))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLVisionBlock(layers.Layer):
    """Pre-norm vision block: LayerNorm + full attention + GELU MLP."""

    def __init__(self, embed_dim, num_heads, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.attn = Qwen3VLVisionAttention(embed_dim, num_heads, name="attn")
        self.mlp = Qwen3VLVisionMLP(embed_dim, intermediate_size, name="mlp")

    def call(self, hidden_states, cos, sin, attention_mask=None):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cos, sin, attention_mask=attention_mask
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLVisionPatchMerger(layers.Layer):
    """Merge 2x2 patches and project to ``out_hidden_size``.

    ``use_postshuffle_norm`` controls whether the LayerNorm runs after the
    merge (DeepStack mergers) or before it (the main merger).
    """

    def __init__(
        self,
        out_hidden_size,
        context_dim,
        spatial_merge_size=2,
        use_postshuffle_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_hidden_size = out_hidden_size
        self.context_dim = context_dim
        self.spatial_merge_size = spatial_merge_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.linear_fc1 = layers.Dense(
            self.hidden_size, use_bias=True, name="linear_fc1"
        )
        self.linear_fc2 = layers.Dense(
            out_hidden_size, use_bias=True, name="linear_fc2"
        )

    def call(self, x):
        if self.use_postshuffle_norm:
            x = self.norm(ops.reshape(x, (-1, self.hidden_size)))
        else:
            x = ops.reshape(self.norm(x), (-1, self.hidden_size))
        return self.linear_fc2(ops.gelu(self.linear_fc1(x), approximate=False))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_hidden_size": self.out_hidden_size,
                "context_dim": self.context_dim,
                "spatial_merge_size": self.spatial_merge_size,
                "use_postshuffle_norm": self.use_postshuffle_norm,
            }
        )
        return config
