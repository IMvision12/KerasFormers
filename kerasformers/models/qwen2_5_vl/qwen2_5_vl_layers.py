"""Qwen2.5-VL layers — deltas vs Qwen2-VL.

The text decoder, attention, and patch embed are unchanged (imported from
``qwen2_vl``). Only the vision block differs: RMSNorm (not LayerNorm) around a
SwiGLU MLP with bias (not the GELU fc1/fc2 MLP). Windowed attention is handled
in the vision model, not here.
"""

import keras
from keras import layers

from kerasformers.models.qwen2_vl.qwen2_vl_layers import (
    Qwen2VLMLP,
    Qwen2VLRMSNorm,
    Qwen2VLVisionAttention,
)


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2_5_VLVisionBlock(layers.Layer):
    """Pre-norm vision block with RMSNorm + SwiGLU MLP (Qwen2.5-VL)."""

    def __init__(self, embed_dim, num_heads, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.norm1 = Qwen2VLRMSNorm(eps=1e-6, name="norm1")
        self.norm2 = Qwen2VLRMSNorm(eps=1e-6, name="norm2")
        self.attn = Qwen2VLVisionAttention(embed_dim, num_heads, name="attn")
        self.mlp = Qwen2VLMLP(embed_dim, intermediate_size, use_bias=True, name="mlp")

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
