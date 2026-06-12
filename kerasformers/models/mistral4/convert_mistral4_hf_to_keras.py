import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "final_norm.weight": "model.norm.weight",
    "decoder_layer_": "model.layers.",
    "attention.q_a_proj": "self_attn.q_a_proj",
    "attention.q_a_layernorm": "self_attn.q_a_layernorm",
    "attention.q_b_proj": "self_attn.q_b_proj",
    "attention.q_proj": "self_attn.q_proj",
    "attention.kv_a_proj_with_mqa": "self_attn.kv_a_proj_with_mqa",
    "attention.kv_a_layernorm": "self_attn.kv_a_layernorm",
    "attention.kv_b_proj": "self_attn.kv_b_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "shared_experts.gate": "shared_experts.gate_proj",
    "shared_experts.up": "shared_experts.up_proj",
    "shared_experts.down": "shared_experts.down_proj",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "mlp.router_weight": "mlp.gate.weight",
    "kernel": "weight",
}


def transfer_mistral4_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        keras_model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = weight.path.split("/", 1)[1].replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
            # Fused routed-expert banks (E, 2I, H) / (E, H, I): direct copy.
            weight.assign(np.asarray(hf_state_dict[name]))
        elif name.endswith("mlp.gate.weight"):
            # Router weight stored (E, H): direct copy (matmul transposes).
            weight.assign(np.asarray(hf_state_dict[name]))
        else:
            transfer_weights(weight.path, weight, hf_state_dict[name])
