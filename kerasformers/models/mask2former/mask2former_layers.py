"""Mask2Former layers: MSDeformAttn, masked attention, sine position embedding.

References:
    - `Masked-attention Mask Transformer for Universal Image Segmentation
      <https://arxiv.org/abs/2112.01527>`_
    - `Deformable DETR: Deformable Transformers for End-to-End Object
      Detection <https://arxiv.org/abs/2010.04159>`_
"""

import math

import keras
import numpy as np
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kerasformers")
class Mask2FormerSinePositionEmbedding(layers.Layer):
    """2D sinusoidal position embedding.

    Generates fixed sine/cosine position encodings for each spatial
    location of an input feature map.

    Input shape: 4D ``(B, H, W, C)``.
    Output shape: 4D ``(B, H, W, hidden_dim)``.
    """

    def __init__(
        self,
        hidden_dim=256,
        temperature=10000,
        normalize=True,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.eps = eps
        self.num_pos_feats = hidden_dim // 2

    def call(self, inputs):
        shape = ops.shape(inputs)
        batch_size = shape[0]
        h = shape[1]
        w = shape[2]

        y_embed = ops.repeat(
            ops.expand_dims(ops.arange(1, h + 1, dtype="float32"), axis=1),
            w,
            axis=1,
        )
        x_embed = ops.repeat(
            ops.expand_dims(ops.arange(1, w + 1, dtype="float32"), axis=0),
            h,
            axis=0,
        )

        if self.normalize:
            y_embed = y_embed / (y_embed[-1:, :] + self.eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, -1:] + self.eps) * 2 * math.pi

        dim_t = ops.arange(self.num_pos_feats, dtype="float32")
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

        pos_x_sin = ops.sin(pos_x[:, :, 0::2])
        pos_x_cos = ops.cos(pos_x[:, :, 1::2])
        pos_x = ops.reshape(
            ops.stack([pos_x_sin, pos_x_cos], axis=-1),
            [h, w, self.num_pos_feats],
        )
        pos_y_sin = ops.sin(pos_y[:, :, 0::2])
        pos_y_cos = ops.cos(pos_y[:, :, 1::2])
        pos_y = ops.reshape(
            ops.stack([pos_y_sin, pos_y_cos], axis=-1),
            [h, w, self.num_pos_feats],
        )

        pos = ops.concatenate([pos_y, pos_x], axis=-1)
        pos = ops.expand_dims(pos, axis=0)
        pos = ops.broadcast_to(pos, [batch_size, h, w, self.hidden_dim])
        return pos

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "hidden_dim": self.hidden_dim,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "eps": self.eps,
            }
        )
        return c


def bilinear_sample(value, sampling_locations, spatial_shapes):
    """Bilinear sample from multi-scale value at sub-pixel locations.

    Args:
        value: ``(B, N_total, n_heads, head_dim)`` flattened multi-scale value tokens.
        sampling_locations: ``(B, N_q, n_heads, n_levels, n_points, 2)`` normalized
            ``[0,1]`` ``(x, y)`` coordinates per level.
        spatial_shapes: list of ``(h, w)`` per level (static Python tuples).

    Returns:
        Sampled features ``(B, N_q, n_heads, n_levels, n_points, head_dim)``.
    """
    b = ops.shape(value)[0]
    n_q = ops.shape(sampling_locations)[1]
    n_heads = value.shape[2]
    head_dim = value.shape[3]
    n_points = sampling_locations.shape[4]

    start = 0
    sampled_per_level = []
    for level_idx, (h, w) in enumerate(spatial_shapes):
        n_level = h * w
        v_level = value[:, start : start + n_level, :, :]
        start += n_level

        loc = sampling_locations[:, :, :, level_idx, :, :]
        # loc: (B, N_q, n_heads, n_points, 2)
        x = loc[..., 0] * float(w) - 0.5
        y = loc[..., 1] * float(h) - 0.5

        x0 = ops.floor(x)
        y0 = ops.floor(y)
        x1 = x0 + 1.0
        y1 = y0 + 1.0

        wx1 = x - x0
        wx0 = 1.0 - wx1
        wy1 = y - y0
        wy0 = 1.0 - wy1

        x0i_raw = ops.cast(x0, "int32")
        x1i_raw = ops.cast(x1, "int32")
        y0i_raw = ops.cast(y0, "int32")
        y1i_raw = ops.cast(y1, "int32")

        # Clip indices to valid range for safe gather; multiply by in-bounds mask
        # at the end to zero out out-of-range contributions ("zeros" padding mode).
        x0i = ops.clip(x0i_raw, 0, w - 1)
        x1i = ops.clip(x1i_raw, 0, w - 1)
        y0i = ops.clip(y0i_raw, 0, h - 1)
        y1i = ops.clip(y1i_raw, 0, h - 1)

        valid_x0 = ops.cast((x0i_raw >= 0) & (x0i_raw < w), "float32")
        valid_x1 = ops.cast((x1i_raw >= 0) & (x1i_raw < w), "float32")
        valid_y0 = ops.cast((y0i_raw >= 0) & (y0i_raw < h), "float32")
        valid_y1 = ops.cast((y1i_raw >= 0) & (y1i_raw < h), "float32")

        v_flat = v_level  # (B, h*w, n_heads, head_dim)

        idx00 = y0i * w + x0i
        idx01 = y0i * w + x1i
        idx10 = y1i * w + x0i
        idx11 = y1i * w + x1i

        def gather_corner(idx):
            # idx: (B, N_q, n_heads, n_points) — transpose so element order
            # in the flat reshape (B, N_q*n_points, n_heads) is (q, p, h).
            idx_t = ops.transpose(idx, (0, 1, 3, 2))  # (B, N_q, n_points, n_heads)
            idx_r = ops.reshape(idx_t, (b, n_q * n_points, n_heads))
            idx_r = ops.expand_dims(idx_r, axis=-1)
            idx_r = ops.broadcast_to(idx_r, (b, n_q * n_points, n_heads, head_dim))
            taken = ops.take_along_axis(v_flat, idx_r, axis=1)
            return ops.reshape(taken, (b, n_q, n_points, n_heads, head_dim))

        v00 = gather_corner(idx00)
        v01 = gather_corner(idx01)
        v10 = gather_corner(idx10)
        v11 = gather_corner(idx11)

        # weights (B, N_q, n_heads, n_points) → broadcast over n_points pos and head_dim:
        # current sampled tensors have layout (B, N_q, n_points, n_heads, head_dim).
        wx0_ = ops.transpose(ops.expand_dims(wx0, axis=-1), (0, 1, 3, 2, 4))
        wx1_ = ops.transpose(ops.expand_dims(wx1, axis=-1), (0, 1, 3, 2, 4))
        wy0_ = ops.transpose(ops.expand_dims(wy0, axis=-1), (0, 1, 3, 2, 4))
        wy1_ = ops.transpose(ops.expand_dims(wy1, axis=-1), (0, 1, 3, 2, 4))
        m_x0 = ops.transpose(ops.expand_dims(valid_x0, axis=-1), (0, 1, 3, 2, 4))
        m_x1 = ops.transpose(ops.expand_dims(valid_x1, axis=-1), (0, 1, 3, 2, 4))
        m_y0 = ops.transpose(ops.expand_dims(valid_y0, axis=-1), (0, 1, 3, 2, 4))
        m_y1 = ops.transpose(ops.expand_dims(valid_y1, axis=-1), (0, 1, 3, 2, 4))

        sampled = (
            v00 * (wx0_ * wy0_ * m_x0 * m_y0)
            + v01 * (wx1_ * wy0_ * m_x1 * m_y0)
            + v10 * (wx0_ * wy1_ * m_x0 * m_y1)
            + v11 * (wx1_ * wy1_ * m_x1 * m_y1)
        )
        # sampled: (B, N_q, n_points, n_heads, head_dim) → (B, N_q, n_heads, n_points, head_dim)
        sampled = ops.transpose(sampled, (0, 1, 3, 2, 4))
        sampled_per_level.append(sampled)

    # Stack along level axis: (B, N_q, n_heads, n_levels, n_points, head_dim)
    return ops.stack(sampled_per_level, axis=3)


@keras.saving.register_keras_serializable(package="kerasformers")
class Mask2FormerDeformableAttention(layers.Layer):
    """Multi-scale deformable attention (MSDeformAttn).

    For each query, learns per-head sampling offsets at each of
    ``n_levels`` multi-scale feature maps and ``n_points`` points per
    level, then computes a weighted sum via predicted attention weights.

    Used in the pixel-decoder MSDeformAttn encoder.

    Args:
        hidden_dim: Model dimension (input & output).
        n_heads: Number of attention heads.
        n_levels: Number of input feature levels.
        n_points: Number of sample points per head per level.

    Inputs (call):
        query: ``(B, N_q, hidden_dim)`` token sequence.
        reference_points: ``(B, N_q, n_levels, 2)`` normalized ref points.
        value: ``(B, N_total, hidden_dim)`` flattened multi-scale features.
        spatial_shapes: list of ``(h, w)`` per level.
    """

    def __init__(self, hidden_dim, n_heads, n_levels, n_points, **kwargs):
        super().__init__(**kwargs)
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})."
            )
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = hidden_dim // n_heads

        self.sampling_offsets = layers.Dense(
            n_heads * n_levels * n_points * 2, name="sampling_offsets"
        )
        self.attention_weights = layers.Dense(
            n_heads * n_levels * n_points, name="attention_weights"
        )
        self.value_proj = layers.Dense(hidden_dim, name="value_proj")
        self.output_proj = layers.Dense(hidden_dim, name="output_proj")

    def call(self, query, reference_points, value, spatial_shapes=None):
        b = ops.shape(query)[0]
        n_q = ops.shape(query)[1]

        value = self.value_proj(value)
        value = ops.reshape(value, (b, -1, self.n_heads, self.head_dim))

        offsets = self.sampling_offsets(query)
        offsets = ops.reshape(
            offsets, (b, n_q, self.n_heads, self.n_levels, self.n_points, 2)
        )

        att = self.attention_weights(query)
        att = ops.reshape(att, (b, n_q, self.n_heads, self.n_levels * self.n_points))
        att = ops.softmax(att, axis=-1)
        att = ops.reshape(att, (b, n_q, self.n_heads, self.n_levels, self.n_points))

        # offset_normalizer: (n_levels, 2) with (w, h) per level
        normalizer = ops.convert_to_tensor(
            [[float(w), float(h)] for h, w in spatial_shapes], dtype="float32"
        )
        # offsets: (B, N_q, n_heads, n_levels, n_points, 2) (last dim is (x, y))
        normalizer = ops.reshape(normalizer, (1, 1, 1, self.n_levels, 1, 2))
        offsets_norm = offsets / normalizer

        # reference_points: (B, N_q, n_levels, 2) -> expand to (B, N_q, 1, n_levels, 1, 2)
        ref = ops.expand_dims(ops.expand_dims(reference_points, axis=2), axis=4)
        sampling_locations = ref + offsets_norm
        # sampling_locations: (B, N_q, n_heads, n_levels, n_points, 2) normalized [0, 1]

        sampled = bilinear_sample(value, sampling_locations, spatial_shapes)
        # sampled: (B, N_q, n_heads, n_levels, n_points, head_dim)

        # weighted sum over (n_levels, n_points)
        att_exp = ops.expand_dims(att, axis=-1)
        out = ops.sum(sampled * att_exp, axis=(3, 4))
        # out: (B, N_q, n_heads, head_dim) → (B, N_q, hidden_dim)
        out = ops.reshape(out, (b, n_q, self.hidden_dim))
        return self.output_proj(out)

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "hidden_dim": self.hidden_dim,
                "n_heads": self.n_heads,
                "n_levels": self.n_levels,
                "n_points": self.n_points,
            }
        )
        return c


@keras.saving.register_keras_serializable(package="kerasformers")
class Mask2FormerCrossAttention(layers.Layer):
    """Cross-attention with PyTorch-style fused QKV input projection.

    Mirrors ``nn.MultiheadAttention``'s ``in_proj_weight`` layout — a
    single ``(3*hidden_dim, hidden_dim)`` matrix that produces Q, K, V
    via slicing — so the converter can transfer ``cross_attn.in_proj_*``
    weights directly. Accepts an additive ``attn_mask`` for masked
    cross-attention.

    Inputs (call):
        query: ``(B, N_q, hidden_dim)``.
        key: ``(B, N_k, hidden_dim)``.
        value: ``(B, N_k, hidden_dim)``.
        attn_mask: optional additive mask broadcastable to
            ``(B*n_heads, N_q, N_k)``.
    """

    def __init__(self, hidden_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.out_proj = layers.Dense(hidden_dim, name="out_proj")

    def build(self, _input_shape):
        self.in_proj_weight = self.add_weight(
            name="in_proj_weight",
            shape=(3 * self.hidden_dim, self.hidden_dim),
            initializer="zeros",
        )
        self.in_proj_bias = self.add_weight(
            name="in_proj_bias",
            shape=(3 * self.hidden_dim,),
            initializer="zeros",
        )
        super().build(_input_shape)

    def project(self, x, slc):
        # x: (B, N, hidden_dim)
        w = self.in_proj_weight[slc[0] : slc[1], :]
        bias = self.in_proj_bias[slc[0] : slc[1]]
        return ops.matmul(x, ops.transpose(w, (1, 0))) + bias

    def call(self, query, key, value, attn_mask=None):
        b = ops.shape(query)[0]
        n_q = ops.shape(query)[1]
        n_k = ops.shape(key)[1]

        d = self.hidden_dim
        q = self.project(query, (0, d))
        k = self.project(key, (d, 2 * d))
        v = self.project(value, (2 * d, 3 * d))

        q = ops.reshape(q, (b, n_q, self.num_heads, self.head_dim))
        k = ops.reshape(k, (b, n_k, self.num_heads, self.head_dim))
        v = ops.reshape(v, (b, n_k, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        if attn_mask is not None:
            # attn_mask: (B, num_heads, N_q, N_k)
            attn = attn + attn_mask

        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, n_q, self.hidden_dim))
        return self.out_proj(out)

    def get_config(self):
        c = super().get_config()
        c.update({"hidden_dim": self.hidden_dim, "num_heads": self.num_heads})
        return c


@keras.saving.register_keras_serializable(package="kerasformers")
class Mask2FormerSelfAttention(layers.Layer):
    """Standard multi-head self-attention with separate q/k/v/out_proj.

    Used in the masked-attention decoder for self-attention over the
    object queries.
    """

    def __init__(self, hidden_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = layers.Dense(hidden_dim, name="q_proj")
        self.k_proj = layers.Dense(hidden_dim, name="k_proj")
        self.v_proj = layers.Dense(hidden_dim, name="v_proj")
        self.out_proj = layers.Dense(hidden_dim, name="out_proj")

    def call(self, query, key, value):
        b = ops.shape(query)[0]
        n_q = ops.shape(query)[1]
        n_k = ops.shape(key)[1]
        q = ops.reshape(self.q_proj(query), (b, n_q, self.num_heads, self.head_dim))
        k = ops.reshape(self.k_proj(key), (b, n_k, self.num_heads, self.head_dim))
        v = ops.reshape(self.v_proj(value), (b, n_k, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, n_q, self.hidden_dim))
        return self.out_proj(out)

    def get_config(self):
        c = super().get_config()
        c.update({"hidden_dim": self.hidden_dim, "num_heads": self.num_heads})
        return c


@keras.saving.register_keras_serializable(package="kerasformers")
class Mask2FormerLearnedEmbedding(layers.Layer):
    """Holds a ``(num_embeddings, hidden_dim)`` learned weight, broadcast over batch.

    HF stores these as ``weight`` (matching ``nn.Embedding``).
    """

    def __init__(self, num_embeddings, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim

    def build(self, _input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(self.num_embeddings, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        super().build(_input_shape)

    def call(self, batch_ref):
        b = ops.shape(batch_ref)[0]
        w = ops.expand_dims(self.weight, axis=0)
        return ops.broadcast_to(w, [b, self.num_embeddings, self.hidden_dim])

    def get_config(self):
        c = super().get_config()
        c.update({"num_embeddings": self.num_embeddings, "hidden_dim": self.hidden_dim})
        return c


@keras.saving.register_keras_serializable(package="kerasformers")
class Mask2FormerReferencePoints(layers.Layer):
    """Generates fixed reference points for the MSDeformAttn pixel encoder.

    Each token gets one ``(x, y)`` per level; the point is the token's
    own ``(x/w, y/h)`` location replicated across all levels. Static
    shape, computed at call time from ``spatial_shapes``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, batch_ref, spatial_shapes=None):
        b = ops.shape(batch_ref)[0]
        n_levels = len(spatial_shapes)
        ref_per_level = []
        for h, w in spatial_shapes:
            ys = (np.arange(h, dtype=np.float32) + 0.5) / float(h)
            xs = (np.arange(w, dtype=np.float32) + 0.5) / float(w)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            ref = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
            ref_per_level.append(ref)
        # ref_all: (N_total, 2)
        ref_all = np.concatenate(ref_per_level, axis=0)
        # Per token, replicate across levels: (N_total, n_levels, 2)
        ref_all_levels = np.tile(ref_all[:, None, :], (1, n_levels, 1))
        t = ops.convert_to_tensor(ref_all_levels)
        t = ops.expand_dims(t, axis=0)
        return ops.broadcast_to(t, (b, ref_all_levels.shape[0], n_levels, 2))
