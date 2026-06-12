import re

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "final_norm.weight": "model.norm.weight",
    "decoder_layer_": "model.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate_weight": "mlp.gate.weight",
    "kernel": "weight",
}


def fuse_expert_weights(hf_state_dict):
    """Canonicalize the hub checkpoints' per-expert MoE layout to the fused one.

    Raw Mixtral checkpoints store ``model.layers.N.block_sparse_moe.gate.weight``
    plus per-expert ``...experts.{e}.w1/w2/w3.weight`` (w1=gate ``(I, H)``,
    w3=up ``(I, H)``, w2=down ``(H, I)``); newer in-memory state dicts already
    carry ``mlp.gate.weight`` / ``mlp.experts.gate_up_proj`` ``(E, 2I, H)`` /
    ``mlp.experts.down_proj`` ``(E, H, I)``. Returns a dict in the fused form.
    """
    if not any("block_sparse_moe" in key for key in hf_state_dict):
        return hf_state_dict
    out = {}
    w1, w3, w2 = {}, {}, {}
    pat = re.compile(
        r"^(model\.layers\.\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$"
    )
    for key, value in hf_state_dict.items():
        match = pat.match(key)
        if match:
            layer, expert, which = match.group(1), int(match.group(2)), match.group(3)
            {"w1": w1, "w3": w3, "w2": w2}[which].setdefault(layer, {})[expert] = value
        elif ".block_sparse_moe.gate." in key:
            out[key.replace(".block_sparse_moe.gate.", ".mlp.gate.")] = value
        else:
            out[key] = value
    for layer in w1:
        experts = sorted(w1[layer])
        gate_up = np.stack(
            [
                np.concatenate(
                    [np.asarray(w1[layer][e]), np.asarray(w3[layer][e])], axis=0
                )
                for e in experts
            ],
            axis=0,
        )  # (E, 2I, H)
        down = np.stack([np.asarray(w2[layer][e]) for e in experts], axis=0)  # (E,H,I)
        out[f"{layer}.mlp.experts.gate_up_proj"] = gate_up
        out[f"{layer}.mlp.experts.down_proj"] = down
    return out


def transfer_mixtral_weights(keras_model, hf_state_dict):
    state = fuse_expert_weights(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        keras_model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = weight.path.split("/", 1)[1].replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
            # Fused per-expert banks (E, 2I, H) / (E, H, I): direct copy.
            weight.assign(np.asarray(state[name]))
        elif name.endswith("mlp.gate.weight"):
            # Router weight stored (E, H): direct copy (matmul transposes).
            weight.assign(np.asarray(state[name]))
        else:
            transfer_weights(weight.path, weight, state[name])
