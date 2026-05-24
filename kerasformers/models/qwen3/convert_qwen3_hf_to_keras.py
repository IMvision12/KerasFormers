"""On-the-fly weight conversion for Qwen3 (HF safetensors -> Keras).

Follows the library's name-mapped convention (see CLIP / DINOv3 / DETR): driven
off the Keras model's own weights, each weight's hierarchical ``path`` is mapped
to the HF tensor name and assigned via the shared ``transfer_weights`` helper.

kerasformers uses its own layer names, so ``hf_weight_name`` bridges them to
HF's. Qwen3 adds per-head q/k RMSNorm and drops the qkv bias.
"""

import numpy as np

from kerasformers.weight_utils.custom_exception import WeightMappingError
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights

# kerasformers layer-name segment -> HF segment (applied after "/"->"."; the
# *_norm entries come first so they aren't shadowed by query/key).
_LAYER_MAP = {
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
}


def hf_weight_name(path):
    """Map a Keras weight ``path`` to its HuggingFace tensor name."""
    rest = path.split("/", 1)[1]  # drop the model-name root
    if rest.startswith("token_embedding"):
        return "model.embed_tokens.weight"
    if rest.startswith("final_norm"):
        return "model.norm.weight"
    if rest.startswith("lm_head"):
        return "lm_head.weight"
    rest = rest.replace("decoder_layer_", "layers.").replace("/", ".")
    for old, new in _LAYER_MAP.items():
        rest = rest.replace(old, new)
    return "model." + rest.replace(".kernel", ".weight")


def build_model(model):
    model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})


def transfer_qwen3_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        build_model(keras_model)
    for weight in keras_model.weights:
        name = hf_weight_name(weight.path)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        transfer_weights(weight.path, weight, hf_state_dict[name])
