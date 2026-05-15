"""timm ResMLP -> Keras weight transfer."""

import re
from typing import Dict

import numpy as np

from kmodels.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kmodels.weight_utils.weight_split_torch_and_keras import split_model_weights
from kmodels.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "stem.conv": "stem.proj",
    "affine.1.alpha": "norm1.alpha",
    "affine.1.beta": "norm1.beta",
    "affine.2.alpha": "norm2.alpha",
    "affine.2.beta": "norm2.beta",
    "dense.1": "linear_tokens",
    "dense.2": "mlp_channels.fc1",
    "dense.3": "mlp_channels.fc2",
    "kernel": "weight",
    "gamma": "weight",
    "Final.affine": "norm",
    "predictions": "head",
}


def transfer_resmlp_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    """Transfer a timm ResMLP state-dict into a Keras :class:`ResMLP`."""
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        torch_weight_name = re.sub(
            r"scale\.(\d+)\.variable(?:\.\d+)?", r"ls\1", torch_weight_name
        )

        if "affine" in keras_weight_name and (
            "alpha" in keras_weight_name or "beta" in keras_weight_name
        ):
            if torch_weight_name not in state_dict:
                raise WeightMappingError(keras_weight_name, torch_weight_name)
            torch_weight = state_dict[torch_weight_name]
            reshaped_weight = torch_weight.reshape(1, 1, -1)
            keras_weight.assign(reshaped_weight)
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
        transfer_weights(keras_weight_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import gc

    import keras

    from kmodels.base.base_model import download_hf_state_dict
    from kmodels.models.resmlp import ResMLPClassify
    from kmodels.models.resmlp.config import RESMLP_WEIGHT_CONFIG

    for variant, meta in RESMLP_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ResMLPClassify.from_weights(variant, load_weights=False)
        transfer_resmlp_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
