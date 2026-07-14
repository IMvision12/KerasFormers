import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

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

VISION_MAPPING = {
    "vision_model.patch_embed": "vision_model.vision_model.embeddings.patch_embedding",
    "vision_model.position_embedding.embeddings": (
        "vision_model.vision_model.embeddings.position_embedding.weight"
    ),
    "vision_model.post_layernorm": "vision_model.vision_model.post_layernorm",
    "vision_model.blocks_": "vision_model.vision_model.encoder.layers.",
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

ALIGNER_MAPPING = {
    "aligner_linear1": "aligner.linear1",
    "aligner_linear2": "aligner.linear2",
    "kernel": "weight",
}


def normalize_keys(hf_state_dict):
    out = {}
    for key, value in hf_state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        out[key] = value
    return out


def transfer_deepseek_vl_weights(keras_model, hf_state_dict):
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
            weight.assign(np.transpose(np.asarray(state[name]), (2, 3, 1, 0)))
        else:
            transfer_weights(weight.path, weight, state[name])


if __name__ == "__main__":
    import gc
    import os

    import keras

    from kerasformers.models.deepseek_vl import DeepseekVLModel
    from kerasformers.models.deepseek_vl.deepseek_vl_config import (
        DEEPSEEK_VL_WEIGHTS_URLS,
    )

    # Only the model_type "deepseek_vl" repos (the 1.3B chat/base) are loadable
    # here. The 7B repos are "deepseek_vl_hybrid" (SAM branch) -- a different
    # architecture -- and are intentionally absent from the config.
    HF_SOURCES = {
        "deepseek_vl_1.3b_chat": "deepseek-community/deepseek-vl-1.3b-chat",
        "deepseek_vl_1.3b_base": "deepseek-community/deepseek-vl-1.3b-base",
    }
    MAX_SHARD_GB = 1.7

    for variant, meta in DEEPSEEK_VL_WEIGHTS_URLS.items():
        hf_id = HF_SOURCES[variant]
        out_path = os.path.basename(meta["url"])
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        model = DeepseekVLModel.from_weights("hf:" + hf_id)

        n_bytes = sum(int(np.prod(w.shape)) * 4 for w in model.weights)
        if out_path.endswith(".json"):
            model.save_weights(out_path, max_shard_size=MAX_SHARD_GB)
        elif n_bytes > 2 * 1024**3:
            raise ValueError(
                f"{variant} is {n_bytes / 1024**3:.2f} GB (> 2 GB); set its config "
                f"URL extension to .weights.json so it shards."
            )
        else:
            model.save_weights(out_path)
        print(f"  Saved -> {out_path}  ({n_bytes / 1024**3:.2f} GB)")

        del model
        keras.backend.clear_session()
        gc.collect()
