import numpy as np

from kerasformers.weight_utils.custom_exception import WeightMappingError
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.language_model.embed_tokens.weight",
    "final_norm.weight": "model.language_model.norm.weight",
    "decoder_layer_": "model.language_model.layers.",
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
    "kernel": "weight",
}


def transfer_qwen3_5_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        keras_model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})
    for weight in keras_model.weights:
        name = weight.path.split("/", 1)[1].replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        torch_weight = hf_state_dict[name]
        if "conv_weight" in weight.path:
            weight.assign(np.asarray(torch_weight).squeeze(1))
        else:
            transfer_weights(weight.path, weight, torch_weight)
