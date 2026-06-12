import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

# The text decoder and lm_head (keras paths starting "language_model." (none
# here — the decoder is inline), "token_embedding", "decoder_layer_",
# "final_norm").
TEXT_MAPPING = {
    "token_embedding.embeddings": "language_model.embed_tokens.weight",
    "final_norm.weight": "language_model.norm.weight",
    "decoder_layer_": "language_model.layers.",
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "post_attention_norm": "post_attention_layernorm",
    "pre_feedforward_norm": "pre_feedforward_layernorm",
    "post_feedforward_norm": "post_feedforward_layernorm",
    "attention_norm": "input_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "kernel": "weight",
}

# The SigLIP tower (keras paths starting "vision_tower.").
VISION_MAPPING = {
    "vision_tower.patch_embed": "vision_tower.vision_model.embeddings.patch_embedding",
    "vision_tower.position_embedding.embeddings": (
        "vision_tower.vision_model.embeddings.position_embedding.weight"
    ),
    "vision_tower.blocks_": "vision_tower.vision_model.encoder.layers.",
    "vision_tower.post_layernorm": "vision_tower.vision_model.post_layernorm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.out_proj",
    "fc1": "mlp.fc1",
    "fc2": "mlp.fc2",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

PROJECTOR_MAPPING = {
    "kernel": "weight",
}


def normalize_keys(hf_state_dict):
    # Text-only checkpoints (gemma3_text) use bare "model.*" keys; multimodal
    # ones use "model.language_model.* / model.vision_tower.* /
    # model.multi_modal_projector.*" (new) or the same without the outer
    # "model." (hub). Canonicalize to "language_model.* / vision_tower.* /
    # multi_modal_projector.* / lm_head.weight".
    keys = list(hf_state_dict.keys())
    has_lm = any("language_model." in key for key in keys)
    out = {}
    for key, value in hf_state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        if not has_lm:
            # text-only layout: layers.* / embed_tokens.* / norm.* at top level
            if not key.startswith(("lm_head.",)):
                key = "language_model." + key
        out[key] = value
    return out


def transfer_gemma3_weights(keras_model, hf_state_dict):
    state = normalize_keys(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        feed = {"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")}
        if keras_model.vision_tower is not None:
            n = keras_model.mm_tokens_per_image
            feed = {
                "input_ids": np.array(
                    [[0] + [keras_model.image_token_id] * n + [1]], dtype="int64"
                ),
                "pixel_values": np.zeros(
                    (1, keras_model.image_size, keras_model.image_size, 3),
                    dtype="float32",
                ),
            }
        keras_model(feed)
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = weight.path.split("/", 1)[1].replace("/", ".")
        if name.startswith("vision_tower."):
            mapping = VISION_MAPPING
        elif name.startswith("multi_modal_projector."):
            mapping = PROJECTOR_MAPPING
        else:
            mapping = TEXT_MAPPING
        for old, new in mapping.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        if name.endswith("patch_embedding.weight"):
            # Conv2D patch embed: HF (out, in, kh, kw) -> Keras (kh, kw, in, out).
            weight.assign(np.transpose(np.asarray(state[name]), (2, 3, 1, 0)))
        elif name.endswith("mm_input_projection_weight"):
            # Raw (vision_dim, text_dim) matrix: direct copy, no transpose.
            weight.assign(np.asarray(state[name]))
        else:
            transfer_weights(weight.path, weight, state[name])
