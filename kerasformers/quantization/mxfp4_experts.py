import keras
from keras import layers, ops

from kerasformers.quantization.mxfp4_quantize import dequantize_mxfp4


@keras.saving.register_keras_serializable(package="kerasformers")
class GptOssMXFP4Experts(layers.Layer):
    """MXFP4-quantized GPT-OSS expert bank, matching the official packed weights.

    The MXFP4 counterpart of ``GptOssExperts``: it lives in the quantization
    package (not the model file) so the model stays quantization-agnostic, the
    way transformers keeps its ``Mxfp4GptOssExperts`` in ``integrations/mxfp4.py``
    rather than ``modeling_gpt_oss.py``.

    Mathematically identical to ``GptOssExperts``, but ``gate_up_proj`` and
    ``down_proj`` are stored in MXFP4 exactly as OpenAI ships them, uint8 nibble
    ``_blocks`` (two 4-bit codebook indices per byte) plus a uint8 e8m0 ``_scales``
    exponent per 32-value block, ~4x smaller than bf16. They are dequantized on the
    fly in ``call`` (weight-only: memory, not speed) and consumed in their natural
    ``(E, 2I, H)`` / ``(E, H, I)`` layout, so no transpose is needed and the block
    tensors map one-to-one onto the checkpoint. Biases stay full precision.

    Args:
        num_experts: Number of experts ``E``.
        embed_dim: Model width ``H`` (must be a multiple of 32).
        mlp_dim: Per-expert hidden width ``I`` (must be a multiple of 32).
    """

    def __init__(self, num_experts, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.alpha = 1.702
        self.limit = 7.0

    def build(self, input_shape):
        e, h, i = self.num_experts, self.embed_dim, self.mlp_dim
        if h % 32 or i % 32:
            raise ValueError(
                f"MXFP4 packs 32-value blocks, so embed_dim ({h}) and mlp_dim ({i}) "
                f"must be multiples of 32."
            )
        # gate_up_proj (E, 2I, H) packed along H; down_proj (E, H, I) packed along I.
        # Shapes match OpenAI's *_blocks / *_scales exactly (direct checkpoint copy).
        self.gate_up_proj_blocks = self.add_weight(
            name="gate_up_proj_blocks",
            shape=(e, 2 * i, h // 32, 16),
            dtype="uint8",
            initializer="zeros",
            trainable=False,
        )
        self.gate_up_proj_scales = self.add_weight(
            name="gate_up_proj_scales",
            shape=(e, 2 * i, h // 32),
            dtype="uint8",
            initializer="zeros",
            trainable=False,
        )
        self.gate_up_proj_bias = self.add_weight(
            name="gate_up_proj_bias", shape=(e, 2 * i), initializer="zeros"
        )
        self.down_proj_blocks = self.add_weight(
            name="down_proj_blocks",
            shape=(e, h, i // 32, 16),
            dtype="uint8",
            initializer="zeros",
            trainable=False,
        )
        self.down_proj_scales = self.add_weight(
            name="down_proj_scales",
            shape=(e, h, i // 32),
            dtype="uint8",
            initializer="zeros",
            trainable=False,
        )
        self.down_proj_bias = self.add_weight(
            name="down_proj_bias", shape=(e, h), initializer="zeros"
        )
        self.built = True

    def call(self, hidden_states, routing_weights):
        dtype = hidden_states.dtype
        gate_up_proj = dequantize_mxfp4(
            self.gate_up_proj_blocks, self.gate_up_proj_scales, dtype
        )  # (E, 2I, H)
        down_proj = dequantize_mxfp4(
            self.down_proj_blocks, self.down_proj_scales, dtype
        )  # (E, H, I)
        gate_up = (
            ops.einsum("th,eih->tei", hidden_states, gate_up_proj)
            + self.gate_up_proj_bias
        )
        gate_up = ops.reshape(gate_up, (-1, self.num_experts, self.mlp_dim, 2))
        gate = ops.minimum(gate_up[..., 0], self.limit)
        up = ops.clip(gate_up[..., 1], -self.limit, self.limit)
        glu = gate * ops.sigmoid(gate * self.alpha)
        gated = (up + 1.0) * glu  # (T, E, I)
        expert_out = (
            ops.einsum("tei,ehi->teh", gated, down_proj) + self.down_proj_bias
        )  # (T, E, H)
        return ops.einsum("te,teh->th", routing_weights, expert_out)  # (T, H)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config
