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
    "post_attention_norm": "post_attention_layernorm",
    "post_feedforward_norm_1": "post_feedforward_layernorm_1",
    "post_feedforward_norm_2": "post_feedforward_layernorm_2",
    "pre_feedforward_norm_2": "pre_feedforward_layernorm_2",
    "pre_feedforward_norm": "pre_feedforward_layernorm",
    "post_feedforward_norm": "post_feedforward_layernorm",
    "attention_norm": "input_layernorm",
    "router.proj": "router.proj",
    "router.scale": "router.scale",
    "router.per_expert_scale": "router.per_expert_scale",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "kernel": "weight",
}


def normalize_keys(hf_state_dict):
    # The omnimodal checkpoints nest the decoder under "model.language_model."
    # (plus audio/vision towers we skip); text-only state dicts use bare
    # "model.*". Canonicalize to "model.*" + "lm_head.weight".
    keys = list(hf_state_dict.keys())
    nested = any(key.startswith("model.language_model.") for key in keys)
    out = {}
    for key, value in hf_state_dict.items():
        if nested:
            if key.startswith("model.language_model."):
                key = "model." + key[len("model.language_model.") :]
            elif key.startswith(
                (
                    "model.audio",
                    "model.vision",
                    "model.embed_audio",
                    "model.embed_vision",
                )
            ):
                continue
        out[key] = value
    return out


def transfer_gemma4_weights(keras_model, hf_state_dict):
    state = normalize_keys(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        keras_model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = weight.path.split("/", 1)[1].replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
            # Fused expert banks (E, 2I, H) / (E, H, I): direct copy.
            weight.assign(np.asarray(state[name]))
        elif name.endswith("router.scale") or name.endswith("router.per_expert_scale"):
            weight.assign(np.asarray(state[name]))
        else:
            transfer_weights(weight.path, weight, state[name])
