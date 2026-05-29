import numpy as np
from tqdm import tqdm

from kerasformers.weight_utils.custom_exception import WeightMappingError
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights

# Applied in order; specific keys before the generic gamma/beta/kernel renames.
WEIGHT_NAME_MAPPING = {
    "wte.embeddings": "wte.weight",
    "wpe.embeddings": "wpe.weight",
    "ln_f.gamma": "ln_f.weight",
    "ln_f.beta": "ln_f.bias",
    "block_": "h.",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

# GPT-2's c_attn/c_proj/c_fc are Conv1D: weight is already (in, out), so it maps
# straight onto the Keras Dense kernel — direct copy, no Linear transpose.
_CONV1D = ("c_attn", "c_proj", "c_fc")


def hf_name_for(path):
    name = path.split("/", 1)[1].replace("/", ".")
    for old, new in WEIGHT_NAME_MAPPING.items():
        name = name.replace(old, new)
    return name


def transfer_gpt2_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        keras_model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = hf_name_for(weight.path)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        if weight.path.endswith("/kernel") and any(c in weight.path for c in _CONV1D):
            weight.assign(np.asarray(hf_state_dict[name]))
        else:
            transfer_weights(weight.path, weight, hf_state_dict[name])
