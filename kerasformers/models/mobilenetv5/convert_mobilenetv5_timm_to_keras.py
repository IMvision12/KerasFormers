import gc
import re
import sys
from typing import Dict

import keras
import numpy as np
from keras import layers
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights
from kerasformers.models.mobilenetv5 import MobileNetV5Encoder

# Hosted variant name -> (arch preset key, timm id, image_size).
MOBILENETV5_VARIANTS = {
    "mobilenetv5_300m_enc": {
        "config": "300m",
        "timm_id": "mobilenetv5_300m_enc",
        "image_size": 768,
    },
}

# Keras top-level layer name -> timm module path.
TOPLEVEL_MAP = {
    "conv_stem_conv": "conv_stem.conv",
    "conv_stem_bn": "conv_stem.bn",
    "msfa_ffn_pw_exp_conv": "msfa.ffn.pw_exp.conv",
    "msfa_ffn_pw_exp_bn": "msfa.ffn.pw_exp.bn",
    "msfa_ffn_pw_proj_conv": "msfa.ffn.pw_proj.conv",
    "msfa_ffn_pw_proj_bn": "msfa.ffn.pw_proj.bn",
    "msfa_norm": "msfa.norm",
}

# Per-block submodule name (the part after ``blocks_{s}_{b}_``) -> timm submodule path.
SUBMODULE_MAP = {
    "conv": "conv",
    "bn1": "bn1",
    "conv_exp": "conv_exp",
    "conv_pwl": "conv_pwl",
    "bn2": "bn2",
    "dw_start_conv": "dw_start.conv",
    "dw_start_bn": "dw_start.bn",
    "pw_exp_conv": "pw_exp.conv",
    "pw_exp_bn": "pw_exp.bn",
    "dw_mid_conv": "dw_mid.conv",
    "dw_mid_bn": "dw_mid.bn",
    "pw_proj_conv": "pw_proj.conv",
    "pw_proj_bn": "pw_proj.bn",
    "layer_scale": "layer_scale",
    "norm": "norm",
    "attn_query_proj": "attn.query.proj",
    "attn_key_down_conv": "attn.key.down_conv",
    "attn_key_norm": "attn.key.norm",
    "attn_key_proj": "attn.key.proj",
    "attn_value_down_conv": "attn.value.down_conv",
    "attn_value_norm": "attn.value.norm",
    "attn_value_proj": "attn.value.proj",
    "attn_output_proj": "attn.output.proj",
}

# Keras variable name -> timm parameter suffix. RmsNorm2d exposes only ``gamma``
# (-> weight); it has no bias / running statistics.
VAR_MAP = {
    "kernel": "weight",
    "bias": "bias",
    "gamma": "weight",
}

_BLOCK_RE = re.compile(r"blocks_(\d+)_(\d+)_(.+)")


def keras_path_to_timm(path: str) -> str:
    """Map a Keras ``weight.path`` (``layer_name/var_name``) to its timm state-dict key."""
    layer_name, var_name = path.rsplit("/", 1)

    match = _BLOCK_RE.match(layer_name)
    if match:
        stage, block, rest = match.group(1), match.group(2), match.group(3)
        if rest not in SUBMODULE_MAP:
            raise WeightMappingError(path, layer_name)
        module_path = f"blocks.{stage}.{block}.{SUBMODULE_MAP[rest]}"
    else:
        if layer_name not in TOPLEVEL_MAP:
            raise WeightMappingError(path, layer_name)
        module_path = TOPLEVEL_MAP[layer_name]

    if module_path.endswith("layer_scale"):
        return f"{module_path}.gamma"
    return f"{module_path}.{VAR_MAP[var_name]}"


def transfer_mobilenetv5_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer timm MobileNetV5 encoder weights into a Keras model by weight path."""
    for layer in tqdm(keras_model.layers, desc="Transferring weights to Keras"):
        if not layer.weights:
            continue
        if isinstance(layer, layers.DepthwiseConv2D):
            hint = "depthwise_conv2d"
        elif isinstance(layer, layers.Conv2D):
            hint = "conv2d"
        else:
            hint = None
        for weight in layer.weights:
            torch_name = keras_path_to_timm(weight.path)
            if torch_name not in state_dict:
                raise WeightMappingError(weight.path, torch_name)
            keras_name = f"{hint}/{weight.name}" if hint else weight.path
            transfer_weights(keras_name, weight, state_dict[torch_name])


if __name__ == "__main__":
    import timm

    sys.setrecursionlimit(10000)

    for variant, meta in MOBILENETV5_VARIANTS.items():
        timm_id = meta["timm_id"]
        size = meta["image_size"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        state = {
            k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()
        }

        keras_model = MobileNetV5Encoder(
            config=meta["config"],
            image_size=size,
            include_normalization=False,
        )

        transfer_mobilenetv5_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
