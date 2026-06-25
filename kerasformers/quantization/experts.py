import keras
from keras import layers, ops

from .fp8_quantize import fp8_supported
from .layers import get_quantizer


@keras.saving.register_keras_serializable(package="kerasformers")
class QuantizedExperts(layers.Layer):
    """Weight-only quantized drop-in for a fused SwiGLU MoE expert bank.

    Replaces a ``...Experts`` layer that stores ``gate_up_proj`` ``(E, 2I, H)``
    and ``down_proj`` ``(E, H, I)`` as fused weights and runs the experts with
    ``einsum``. Both banks are quantized along their **contracting (last) axis**
    via the shared int8 / int4 / fp8 quantizers and dequantized on the fly.
    ``activation`` is the gate nonlinearity (``"silu"`` for most MoE LLMs,
    ``"gelu"`` for Gemma-style).
    """

    def __init__(
        self,
        num_experts,
        embed_dim,
        mlp_dim,
        mode="int8",
        group_size=32,
        activation="silu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mode == "fp8" and not fp8_supported():
            raise ValueError(
                "fp8 quantization requires the torch or jax backend "
                "(tensorflow lacks float8 casts)."
            )
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.mode = mode
        self.group_size = group_size
        self.activation = activation
        self.quantizer = get_quantizer(mode, group_size)

    def build(self, input_shape=None):
        e, i, h = self.num_experts, self.mlp_dim, self.embed_dim
        gate_up = self.quantizer.storage_spec((e, 2 * i, h), axis=-1)
        down = self.quantizer.storage_spec((e, h, i), axis=-1)
        self.gate_up_q = self.add_weight(
            name="gate_up_proj",
            shape=gate_up["kernel"][0],
            dtype=gate_up["kernel"][1],
            initializer="zeros",
            trainable=False,
        )
        self.gate_up_scale = self.add_weight(
            name="gate_up_scale",
            shape=gate_up["scale"][0],
            dtype=gate_up["scale"][1],
            initializer="ones",
            trainable=False,
        )
        self.down_q = self.add_weight(
            name="down_proj",
            shape=down["kernel"][0],
            dtype=down["kernel"][1],
            initializer="zeros",
            trainable=False,
        )
        self.down_scale = self.add_weight(
            name="down_scale",
            shape=down["scale"][0],
            dtype=down["scale"][1],
            initializer="ones",
            trainable=False,
        )
        self.built = True

    def call(self, hidden_states, routing_weights):
        act = ops.gelu if self.activation == "gelu" else ops.silu
        gate_up_w = self.quantizer.dequantize(
            self.gate_up_q, self.gate_up_scale, axis=-1, dtype=hidden_states.dtype
        )
        gate_up = ops.einsum("th,eoh->teo", hidden_states, gate_up_w)
        gate = gate_up[..., : self.mlp_dim]
        up = gate_up[..., self.mlp_dim :]
        down_w = self.quantizer.dequantize(
            self.down_q, self.down_scale, axis=-1, dtype=hidden_states.dtype
        )
        expert_out = ops.einsum("tei,ehi->teh", act(gate) * up, down_w)
        return ops.einsum("te,teh->th", routing_weights, expert_out)

    @classmethod
    def from_experts(cls, experts, mode, group_size=32, activation="silu"):
        layer = cls(
            experts.num_experts,
            experts.embed_dim,
            experts.mlp_dim,
            mode=mode,
            group_size=group_size,
            activation=activation,
            name=experts.name,
        )
        layer.build()
        gate_up_q, gate_up_scale = layer.quantizer.quantize(
            experts.gate_up_proj, axis=-1
        )
        down_q, down_scale = layer.quantizer.quantize(experts.down_proj, axis=-1)
        layer.gate_up_q.assign(gate_up_q)
        layer.gate_up_scale.assign(gate_up_scale)
        layer.down_q.assign(down_q)
        layer.down_scale.assign(down_scale)
        return layer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "mode": self.mode,
                "group_size": self.group_size,
                "activation": self.activation,
            }
        )
        return config
