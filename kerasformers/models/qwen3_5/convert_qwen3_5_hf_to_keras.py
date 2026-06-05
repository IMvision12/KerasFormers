import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "final_norm.weight": "model.norm.weight",
    "decoder_layer_": "model.layers.",
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


def split_gated_deltanet_in_proj(layer, qkvz, ba):
    nk, nv = layer.num_k_heads, layer.num_v_heads
    hk, hv = layer.head_k_dim, layer.head_v_dim
    ratio = nv // nk
    hidden = qkvz.shape[1]
    group = 2 * hk + 2 * ratio * hv
    qkvz = qkvz.reshape(nk, group, hidden)
    q = qkvz[:, :hk].reshape(nk * hk, hidden)
    k = qkvz[:, hk : 2 * hk].reshape(nk * hk, hidden)
    v = qkvz[:, 2 * hk : 2 * hk + ratio * hv].reshape(nv * hv, hidden)
    z = qkvz[:, 2 * hk + ratio * hv :].reshape(nv * hv, hidden)
    ba = ba.reshape(nk, 2 * ratio, hidden)
    b = ba[:, :ratio].reshape(nv, hidden)
    a = ba[:, ratio:].reshape(nv, hidden)
    return np.concatenate([q, k, v], axis=0), z, b, a


def transfer_qwen3_5_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        keras_model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})

    handled = set()
    for i, decoder_layer in enumerate(keras_model.decoder_layers):
        if getattr(decoder_layer, "layer_type", None) == "full_attention":
            continue
        gdn = decoder_layer.linear_attn
        prefix = f"model.layers.{i}.linear_attn"
        qkvz = np.asarray(hf_state_dict[f"{prefix}.in_proj_qkvz.weight"])
        ba = np.asarray(hf_state_dict[f"{prefix}.in_proj_ba.weight"])
        qkv_w, z_w, b_w, a_w = split_gated_deltanet_in_proj(gdn, qkvz, ba)
        for dense, packed in (
            (gdn.in_proj_qkv, qkv_w),
            (gdn.in_proj_z, z_w),
            (gdn.in_proj_b, b_w),
            (gdn.in_proj_a, a_w),
        ):
            transfer_weights(dense.kernel.path, dense.kernel, packed)
            handled.add(dense.kernel.path)

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        if weight.path in handled:
            continue
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
