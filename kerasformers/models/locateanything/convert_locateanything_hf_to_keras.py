import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "vision_model.patch_proj": "vision_model.patch_embed.proj",
    "vision_model.pos_emb": "vision_model.patch_embed.pos_emb.weight",
    "vision_model.final_norm": "vision_model.encoder.final_layernorm",
    "block_": "encoder.blocks.",
    "mlp1_norm": "mlp1.0",
    "mlp1_fc1": "mlp1.1",
    "mlp1_fc2": "mlp1.3",
    "token_embedding.embeddings": "language_model.model.embed_tokens.weight",
    "final_norm.weight": "language_model.model.norm.weight",
    "decoder_layer_": "language_model.model.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}


def hf_name_for(path):
    for old, new in WEIGHT_NAME_MAPPING.items():
        path = path.replace(old, new)
    return path


def build_for_transfer(keras_model):
    grid = np.array([[2, 2]], dtype="int64")
    pixel_values = np.zeros((4, 3, 14, 14), dtype="float32")
    img = keras_model.image_token_index
    input_ids = np.array([[img, 0, 0, 0]], dtype="int64")
    keras_model(
        {"input_ids": input_ids, "pixel_values": pixel_values, "image_grid_hws": grid}
    )


def transfer_locateanything_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        build_for_transfer(keras_model)
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        path = weight.path.split("/", 1)[1].replace("/", ".")
        name = hf_name_for(path)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        value = hf_state_dict[name]
        if path.endswith("patch_proj.kernel"):
            weight.assign(np.transpose(np.asarray(value), (2, 3, 1, 0)))
        elif path.endswith("pos_emb"):
            weight.assign(np.asarray(value))
        else:
            transfer_weights(weight.path, weight, value)


def safetensors_state_dict(files):
    from safetensors import safe_open

    handles = {}
    for f in files:
        fh = safe_open(f, framework="pt")
        for k in fh.keys():
            handles[k] = fh

    class _View:
        def __contains__(self, k):
            return k in handles

        def __getitem__(self, k):
            return handles[k].get_tensor(k).float().cpu().numpy()

    return _View()


if __name__ == "__main__":
    import gc
    import glob
    import json
    import os

    import keras
    from huggingface_hub import snapshot_download

    from kerasformers.models.locateanything import LocateAnythingGenerate
    from kerasformers.models.locateanything.config import LOCATEANYTHING_WEIGHTS_URLS

    DTYPE = "bfloat16"
    MAX_SHARD_GB = 1.7
    HF_SOURCES = {"locateanything_3b": "nvidia/LocateAnything-3B"}

    keras.config.set_dtype_policy(DTYPE)

    for variant, meta in LOCATEANYTHING_WEIGHTS_URLS.items():
        hf_id = HF_SOURCES[variant]
        print(
            f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}  ({DTYPE})\n{'=' * 60}"
        )

        local = snapshot_download(
            hf_id, allow_patterns=["*.json", "*.txt", "*.safetensors"]
        )

        with open(os.path.join(local, "config.json")) as f:
            hf_config = json.load(f)
        model = LocateAnythingGenerate(
            **LocateAnythingGenerate.config_from_hf(hf_config)
        )
        shards = sorted(glob.glob(os.path.join(local, "*.safetensors")))
        transfer_locateanything_weights(model, safetensors_state_dict(shards))

        weights_path = os.path.basename(meta["url"])
        if weights_path.endswith(".json"):
            model.save_weights(weights_path, max_shard_size=MAX_SHARD_GB)
        else:
            model.save_weights(weights_path)
        print(f"  Saved weights -> {weights_path}")

        del model
        keras.backend.clear_session()
        gc.collect()
