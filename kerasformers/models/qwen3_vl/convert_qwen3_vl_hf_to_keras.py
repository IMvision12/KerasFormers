"""On-the-fly weight conversion for Qwen3-VL (HF safetensors -> Keras).

Name-mapped transfer (see CLIP / DINOv3 / DETR), reusing Qwen2-VL's
``normalize_state`` / ``build_model``. Qwen3-VL adds a learned vision
``pos_embed`` (assigned directly), a Conv3d patch embed with bias, ``linear_fc``
vision MLPs, DeepStack mergers (``deepstack_merger_list``), and a Qwen3 text
decoder (per-head q/k RMSNorm, no qkv bias).
"""

import numpy as np

from kerasformers.models.qwen2_vl.convert_qwen2_vl_hf_to_keras import (
    build_model,
    normalize_state,
)
from kerasformers.weight_utils.custom_exception import WeightMappingError
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights


def hf_weight_name(path):
    """Map a Keras weight ``path`` to its HuggingFace (legacy-layout) name."""
    rest = path.split("/", 1)[1]  # drop the model-name root
    if rest.startswith("visual/"):
        name = rest.replace("/", ".").replace("blocks_", "blocks.")
        name = name.replace("deepstack_merger_", "deepstack_merger_list.")
        name = name.replace(".gamma", ".weight").replace(".beta", ".bias")
        if name.endswith("pos_embed"):
            return name + ".weight"
        return name.replace(".kernel", ".weight")
    if rest.startswith("embed_tokens/"):
        return "model.embed_tokens.weight"
    if rest.startswith("lm_head"):
        return "lm_head.weight"
    name = (
        rest[len("language_model/") :].replace("/", ".").replace("layers_", "layers.")
    )
    return "model." + name.replace(".kernel", ".weight")


def transfer_qwen3_vl_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        build_model(keras_model)
    state = normalize_state(hf_state_dict)
    for weight in keras_model.weights:
        name = hf_weight_name(weight.path)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        torch_weight = state[name]
        if weight.path.endswith("pos_embed"):
            weight.assign(
                np.asarray(torch_weight)
            )  # (num_pos, embed_dim), no transpose
        elif "patch_embed" in weight.path and weight.path.endswith("kernel"):
            # Conv3d (embed_dim, in, t, p, p) -> Dense (in*t*p*p, embed_dim)
            tw = np.asarray(torch_weight)
            transfer_weights(weight.path, weight, tw.reshape(tw.shape[0], -1))
        else:
            transfer_weights(weight.path, weight, torch_weight)
