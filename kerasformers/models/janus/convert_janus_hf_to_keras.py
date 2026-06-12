import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

# The Llama text decoder + lm_head (keras paths "decoder_layer_*", the bare
# "token_embedding" and "final_norm", and "lm_head").
TEXT_MAPPING = {
    "token_embedding.embeddings": "language_model.embed_tokens.weight",
    "final_norm.weight": "language_model.norm.weight",
    "decoder_layer_": "language_model.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "kernel": "weight",
}

# The SigLIP-style tower (keras paths "vision_model.*"); the attention output
# projection is named "projection_layer" in HF Janus.
VISION_MAPPING = {
    "vision_model.patch_embed": "vision_model.embeddings.patch_embedding",
    "vision_model.position_embedding.embeddings": (
        "vision_model.embeddings.position_embedding.weight"
    ),
    "vision_model.post_layernorm": "vision_model.post_layernorm",
    "vision_model.blocks_": "vision_model.encoder.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.projection_layer",
    "fc1": "mlp.fc1",
    "fc2": "mlp.fc2",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

ALIGNER_MAPPING = {
    "aligner_fc1": "aligner.fc1",
    "aligner_hidden": "aligner.hidden_layers.0",
    "kernel": "weight",
}


def normalize_keys(hf_state_dict):
    # JanusForConditionalGeneration prefixes everything except lm_head with
    # "model."; strip it. The VQ-VAE image-generation stack (vqmodel,
    # generation_embeddings / generation_aligner / generation_head) is not
    # ported — those keys simply stay unused.
    out = {}
    for key, value in hf_state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        out[key] = value
    return out


def transfer_janus_weights(keras_model, hf_state_dict):
    state = normalize_keys(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        size = keras_model.image_size
        n_tokens = (size // keras_model.patch_size) ** 2
        keras_model(
            {
                "input_ids": np.array(
                    [[0] + [keras_model.image_token_id] * n_tokens + [1]],
                    dtype="int64",
                ),
                "pixel_values": np.zeros((1, size, size, 3), dtype="float32"),
            }
        )
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = weight.path.split("/", 1)[1].replace("/", ".")
        if name.startswith("vision_model."):
            mapping = VISION_MAPPING
        elif name.startswith("aligner_"):
            mapping = ALIGNER_MAPPING
        else:
            mapping = TEXT_MAPPING
        for old, new in mapping.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        if name.endswith("patch_embedding.weight"):
            # Conv2D patch embed: HF (out, in, kh, kw) -> Keras (kh, kw, in, out).
            weight.assign(np.transpose(np.asarray(state[name]), (2, 3, 1, 0)))
        else:
            transfer_weights(weight.path, weight, state[name])
