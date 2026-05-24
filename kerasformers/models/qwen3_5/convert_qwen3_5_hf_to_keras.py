"""On-the-fly weight conversion for Qwen3.5 (HF safetensors -> Keras).

Follows the library's name-mapped convention (see CLIP / DINOv3 / DETR): driven
off the Keras model's own weights, each weight's hierarchical ``path`` is mapped
to the HF tensor name and assigned via the shared ``transfer_weights`` helper.

Qwen3.5 ships multimodal; the text tower lives under ``model.language_model.*``
(vision under ``model.visual.*`` and an ``mtp.*`` head are ignored). kerasformers
uses its own layer names, so ``hf_weight_name`` bridges them to HF's. The one
tensor needing manual handling is the Gated-DeltaNet depthwise ``conv1d`` weight
(``(conv_dim, 1, k)`` -> ``(conv_dim, k)``).
"""

import numpy as np

from kerasformers.weight_utils.custom_exception import WeightMappingError
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights

HF_PREFIX = "model.language_model."

# kerasformers layer-name segment -> HF segment (applied after "/"->"."; the
# *_norm entries precede query/key so they aren't shadowed).
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
    "linear_attn.conv_weight": "linear_attn.conv1d.weight",
}


def hf_weight_name(path):
    """Map a Keras weight ``path`` to its HuggingFace tensor name."""
    rest = path.split("/", 1)[1]  # drop the model-name root
    if rest.startswith("token_embedding"):
        return HF_PREFIX + "embed_tokens.weight"
    if rest.startswith("final_norm"):
        return HF_PREFIX + "norm.weight"
    if rest.startswith("lm_head"):
        return "lm_head.weight"
    rest = rest.replace("decoder_layer_", "layers.").replace("/", ".")
    for old, new in _LAYER_MAP.items():
        rest = rest.replace(old, new)
    return HF_PREFIX + rest.replace(".kernel", ".weight")


def build_model(model):
    model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})


def transfer_qwen3_5_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        build_model(keras_model)
    for weight in keras_model.weights:
        name = hf_weight_name(weight.path)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        torch_weight = hf_state_dict[name]
        if "conv_weight" in weight.path:
            # depthwise causal conv1d: (conv_dim, 1, kernel) -> (conv_dim, kernel)
            weight.assign(np.asarray(torch_weight).squeeze(1))
        else:
            transfer_weights(weight.path, weight, torch_weight)
