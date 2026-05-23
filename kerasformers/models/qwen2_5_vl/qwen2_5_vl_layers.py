"""Pure Keras 3 layers for Qwen2.5-VL (self-contained — no qwen2_vl imports).

Text decoder layers match Qwen2 (RMSNorm, GQA with q/k/v bias + bias-free
o_proj, SwiGLU). The vision tower differs from Qwen2-VL: RMSNorm blocks with a
SwiGLU MLP (bias) and an RMSNorm ``ln_q`` in the patch merger. Windowed
attention is orchestrated by the model, not here.
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
class Qwen2_5_VLRMSNorm(layers.Layer):
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
class Qwen2_5_VLMLP(layers.Layer):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` (text bias-free, vision biased)."""

    def __init__(self, hidden_size, intermediate_size, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_bias = use_bias
        self.gate_proj = layers.Dense(
            intermediate_size, use_bias=use_bias, name="gate_proj"
        )
        self.up_proj = layers.Dense(
            intermediate_size, use_bias=use_bias, name="up_proj"
        )
        self.down_proj = layers.Dense(hidden_size, use_bias=use_bias, name="down_proj")

    def call(self, x):
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "use_bias": self.use_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5_VLAttention(layers.Layer):
    """Grouped-query causal self-attention with multimodal rotary positions."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = layers.Dense(
            num_attention_heads * self.head_dim, use_bias=True, name="q_proj"
        )
        self.k_proj = layers.Dense(
            num_key_value_heads * self.head_dim, use_bias=True, name="k_proj"
        )
        self.v_proj = layers.Dense(
            num_key_value_heads * self.head_dim, use_bias=True, name="v_proj"
        )
        self.o_proj = layers.Dense(hidden_size, use_bias=False, name="o_proj")

    def _split_heads(self, x, num_heads):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        return ops.transpose(
            ops.reshape(x, (b, s, num_heads, self.head_dim)), (0, 2, 1, 3)
        )

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
        query = self._split_heads(self.q_proj(hidden_states), self.num_attention_heads)
        key = self._split_heads(self.k_proj(hidden_states), self.num_key_value_heads)
        value = self._split_heads(self.v_proj(hidden_states), self.num_key_value_heads)

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            key = ops.concatenate([past_k, key], axis=2)
            value = ops.concatenate([past_v, value], axis=2)
        new_key_value = (key, value) if use_cache else None

        if self.num_key_value_groups > 1:
            key = ops.repeat(key, self.num_key_value_groups, axis=1)
            value = ops.repeat(value, self.num_key_value_groups, axis=1)

        attn = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, value)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, q_len, self.num_attention_heads * self.head_dim))
        out = self.o_proj(out)
        return (out, new_key_value) if use_cache else out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5_VLDecoderLayer(layers.Layer):
    """One Qwen2 decoder block: pre-norm GQA attention then pre-norm SwiGLU."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim=None,
        rms_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.input_layernorm = Qwen2_5_VLRMSNorm(
            eps=rms_norm_eps, name="input_layernorm"
        )
        self.self_attn = Qwen2_5_VLAttention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim=self.head_dim,
            name="self_attn",
        )
        self.post_attention_layernorm = Qwen2_5_VLRMSNorm(
            eps=rms_norm_eps, name="post_attention_layernorm"
        )
        self.mlp = Qwen2_5_VLMLP(hidden_size, intermediate_size, name="mlp")

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
        new_key_value = None
        if use_cache:
            attn_out, new_key_value = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_key_value) if use_cache else hidden_states

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
class Qwen2_5_VisionPatchEmbed(layers.Layer):
    """Per-patch linear projection (HF's kernel==stride Conv3d as a Dense)."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.proj = layers.Dense(embed_dim, use_bias=False, name="proj")

    def call(self, x):
        return self.proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5_VLVisionAttention(layers.Layer):
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
class Qwen2_5_VLVisionBlock(layers.Layer):
    """Pre-norm vision block with RMSNorm + SwiGLU MLP (Qwen2.5-VL)."""

    def __init__(self, embed_dim, num_heads, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.norm1 = Qwen2_5_VLRMSNorm(eps=1e-6, name="norm1")
        self.norm2 = Qwen2_5_VLRMSNorm(eps=1e-6, name="norm2")
        self.attn = Qwen2_5_VLVisionAttention(embed_dim, num_heads, name="attn")
        self.mlp = Qwen2_5_VLMLP(
            embed_dim, intermediate_size, use_bias=True, name="mlp"
        )

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
class Qwen2_5_VLPatchMerger(layers.Layer):
    """Merge each 2x2 patch group and project to the LLM hidden size.

    ``ln_q`` is an RMSNorm over the vision dim (Qwen2.5-VL); ``gelu`` is exact.
    """

    def __init__(self, dim, context_dim, spatial_merge_size=2, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.context_dim = context_dim
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2_5_VLRMSNorm(eps=1e-6, name="ln_q")
        self.mlp_fc1 = layers.Dense(self.hidden_size, use_bias=True, name="mlp_fc1")
        self.mlp_fc2 = layers.Dense(dim, use_bias=True, name="mlp_fc2")

    def call(self, x):
        x = ops.reshape(self.ln_q(x), (-1, self.hidden_size))
        return self.mlp_fc2(ops.gelu(self.mlp_fc1(x), approximate=False))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "context_dim": self.context_dim,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config
