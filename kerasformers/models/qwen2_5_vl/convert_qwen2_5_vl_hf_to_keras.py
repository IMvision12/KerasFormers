"""On-the-fly weight conversion for Qwen2.5-VL (HF safetensors -> Keras).

Qwen2.5-VL shares Qwen2-VL's HF key layout one-for-one: the text decoder is
identical, and although the vision tower swaps LayerNorm/GELU for RMSNorm/SwiGLU
the *weight names* line up (RMSNorm ``weight`` and the SwiGLU ``gate/up/down``
map directly, handled by the shared ``transfer_weights``). So the conversion
reuses Qwen2-VL's name-mapped transfer verbatim.
"""

from kerasformers.models.qwen2_vl.convert_qwen2_vl_hf_to_keras import (
    transfer_qwen2_vl_weights,
)


def transfer_qwen2_5_vl_weights(keras_model, hf_state_dict):
    transfer_qwen2_vl_weights(keras_model, hf_state_dict)
