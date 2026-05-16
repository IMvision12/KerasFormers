"""timm MaxViT -> Keras weight transfer."""

from typing import Dict

import numpy as np

from kerasformers.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kerasformers.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "relative_position_bias_table": "RPBT",
    "moving_variance": "MOVVAR",
    "moving_mean": "MOVMEAN",
    "attn_block": "ATTNBLOCK",
    "attn_grid": "ATTNGRID",
    "shortcut_expand": "SHORTCUTEXPAND",
    "pre_logits": "PRELOGITS",
    "pre_norm": "PRENORM",
    "conv1_1x1": "CONV11X1",
    "conv2_kxk": "CONV2KXK",
    "conv3_1x1": "CONV31X1",
    "rel_pos": "RELPOS",
    "se_fc": "SEFC",
    "attn_qkv": "ATTNQKV",
    "attn_proj": "ATTNPROJ",
    "mlp_fc": "MLPFC",
    "_": ".",
    "RPBT": "relative_position_bias_table",
    "MOVVAR": "running_var",
    "MOVMEAN": "running_mean",
    "ATTNBLOCK": "attn_block",
    "ATTNGRID": "attn_grid",
    "SHORTCUTEXPAND": "shortcut.expand",
    "PRELOGITS": "pre_logits",
    "PRENORM": "pre_norm",
    "CONV11X1": "conv1_1x1",
    "CONV2KXK": "conv2_kxk",
    "CONV31X1": "conv3_1x1",
    "RELPOS": "rel_pos",
    "SEFC": "conv.se.fc",
    "ATTNQKV": "attn.qkv",
    "ATTNPROJ": "attn.proj",
    "MLPFC": "mlp.fc",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def transfer_maxvit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    """Transfer a timm MaxViT state-dict into a Keras :class:`MaxViT`."""
    all_keras_weights = []
    for layer in keras_model.layers:
        for w in layer.weights:
            path = w.path
            parts = path.split("/")
            layer_name = parts[-2] if len(parts) >= 2 else parts[0]
            weight_suffix = parts[-1]
            keras_weight_name = f"{layer_name}_{weight_suffix}"
            all_keras_weights.append((w, keras_weight_name))

    for keras_weight, keras_weight_name in all_keras_weights:
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]
        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_weight_name, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight_name,
                keras_weight.shape,
                torch_weight_name,
                torch_weight.shape,
            )

        transfer_name = keras_weight_name
        if "conv2_kxk" in keras_weight_name:
            transfer_name = "dwconv_" + keras_weight_name
        elif "se_fc" in keras_weight_name:
            transfer_name = "conv_" + keras_weight_name
        transfer_weights(transfer_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import gc

    import keras

    from kerasformers.base.base_model import download_hf_state_dict
    from kerasformers.models.maxvit import MaxViTClassify as MaxViT
    from kerasformers.models.maxvit.config import MAXVIT_WEIGHT_CONFIG

    for variant, meta in MAXVIT_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = MaxViT.from_weights(variant, load_weights=False)
        transfer_maxvit_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
