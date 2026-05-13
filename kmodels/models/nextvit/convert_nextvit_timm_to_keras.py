"""timm NextViT -> Keras weight transfer."""

from typing import Dict

import numpy as np

from kmodels.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kmodels.weight_utils.weight_split_torch_and_keras import split_model_weights
from kmodels.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "e.mhsa": "e_mhsa",
    "group.conv3x3": "group_conv3x3",
    "patch.embed": "patch_embed",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

_E_MHSA_NAME_REPLACEMENTS: Dict[str, str] = {"e.mhsa": "e_mhsa"}


def transfer_nextvit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    """Transfer a timm NextViT state-dict into a Keras :class:`NextViT`."""
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "e_mhsa" in keras_weight_name:
            transfer_attention_weights(
                keras_weight_name,
                keras_weight,
                state_dict,
                name_replacements=_E_MHSA_NAME_REPLACEMENTS,
            )
            continue

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
        if len(keras_weight.shape) == 4 and "conv" not in keras_weight_name.lower():
            transfer_name = "conv_" + keras_weight_name
        transfer_weights(transfer_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import gc

    import keras

    from kmodels.base.base_model import download_hf_state_dict
    from kmodels.models.nextvit import NextViTClassify as NextViT
    from kmodels.models.nextvit.config import NEXTVIT_CONFIG

    for variant, cfg in NEXTVIT_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = NextViT.from_weights(variant, load_weights=False)
        transfer_nextvit_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
