import gc
import re
import sys
from typing import Dict

import keras
import numpy as np
from keras import layers
from tqdm import tqdm

from kerasformers.conversion import verify_cls_model_equivalence
from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights
from kerasformers.models.mobilenetv4 import MobileNetV4ImageClassify

# Architecture presets, kept here (not in the package config): models load by Hub
# repo id / kf_config.json. Only this converter builds an untrained model to
# transfer timm weights into.
MOBILENETV4_MODEL_CONFIG = {
    "mobilenetv4_conv_small": {"config": "conv_small", "image_size": 224},
    "mobilenetv4_conv_medium": {"config": "conv_medium", "image_size": 256},
    "mobilenetv4_conv_large": {"config": "conv_large", "image_size": 384},
    "mobilenetv4_hybrid_medium": {"config": "hybrid_medium", "image_size": 224},
    "mobilenetv4_hybrid_large": {"config": "hybrid_large", "image_size": 384},
}

# Hosted variant name -> (arch preset key, timm id). Weights load by Hub repo id.
MOBILENETV4_VARIANTS = {
    "mobilenetv4_conv_small_e2400_r224_in1k": {
        "model": "mobilenetv4_conv_small",
        "timm_id": "mobilenetv4_conv_small.e2400_r224_in1k",
    },
    "mobilenetv4_conv_medium_e500_r256_in1k": {
        "model": "mobilenetv4_conv_medium",
        "timm_id": "mobilenetv4_conv_medium.e500_r256_in1k",
    },
    "mobilenetv4_conv_large_e600_r384_in1k": {
        "model": "mobilenetv4_conv_large",
        "timm_id": "mobilenetv4_conv_large.e600_r384_in1k",
    },
    "mobilenetv4_hybrid_medium_e500_r224_in1k": {
        "model": "mobilenetv4_hybrid_medium",
        "timm_id": "mobilenetv4_hybrid_medium.e500_r224_in1k",
    },
    "mobilenetv4_hybrid_large_e600_r384_in1k": {
        "model": "mobilenetv4_hybrid_large",
        "timm_id": "mobilenetv4_hybrid_large.e600_r384_in1k",
    },
}

# Keras top-level layer name -> timm module path.
TOPLEVEL_MAP = {
    "conv_stem": "conv_stem",
    "bn1": "bn1",
    "conv_head": "conv_head",
    "norm_head": "norm_head",
    "classifier": "classifier",
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

# Keras variable name -> timm parameter suffix (BN gamma/beta -> weight/bias).
VAR_MAP = {
    "kernel": "weight",
    "bias": "bias",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
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
        # timm LayerScale2d exposes a single ``gamma`` parameter.
        return f"{module_path}.gamma"
    return f"{module_path}.{VAR_MAP[var_name]}"


def transfer_mobilenetv4_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer timm MobileNetV4 weights into a Keras model by mapping weight paths.

    Iterates the model's layers so the conv type (regular vs depthwise) is known
    from the layer class, and passes an explicit transpose hint to
    :func:`transfer_weights` (whose transpose branch is name-driven).
    """
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

    for variant, meta in MOBILENETV4_VARIANTS.items():
        model_cfg = dict(MOBILENETV4_MODEL_CONFIG[meta["model"]])
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        state = {
            k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()
        }
        num_classes = int(state["classifier.weight"].shape[0])

        keras_model = MobileNetV4ImageClassify(
            config=model_cfg["config"],
            image_size=model_cfg["image_size"],
            num_classes=num_classes,
            include_normalization=False,
        )

        transfer_mobilenetv4_weights(keras_model, state)

        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="torch_to_keras",
            run_performance=False,
            atol=1e-4,
            rtol=1e-4,
        )
        if not results["standard_input"]:
            raise ValueError(f"{variant}: model equivalence test failed")

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
