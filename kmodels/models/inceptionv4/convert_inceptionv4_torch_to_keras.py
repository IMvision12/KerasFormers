"""timm InceptionV4 -> Keras weight transfer."""

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
    "features_": "features.",
    "_conv": ".conv",
    "_kernel": ".weight",
    "_gamma": ".weight",
    "_beta": ".bias",
    "_bias": ".bias",
    "_bn": ".bn",
    "_moving_mean": ".running_mean",
    "_moving_variance": ".running_var",
    "predictions": "last_linear",
}


def transfer_inceptionv4_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer a timm InceptionV4 state-dict into a Keras :class:`InceptionV4`."""
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        torch_weight_name = re.sub(
            r"features\.(\d+)_", r"features.\1.", torch_weight_name
        )

        if re.match(r"features\.(19|20|21)\.", torch_weight_name):
            if "branch3_1" in torch_weight_name:
                torch_weight_name = torch_weight_name.replace("branch3_1", "branch3.1")
            else:
                torch_weight_name = re.sub(
                    r"\.branch([12])\.([0-9]+[ab]?)",
                    r".branch\1_\2",
                    torch_weight_name,
                )
        else:
            torch_weight_name = re.sub(
                r"\.branch([0-9])_([0-9][ab]?)",
                r".branch\1.\2",
                torch_weight_name,
            )
            torch_weight_name = torch_weight_name.replace("branch3_1", "branch3.1")

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

    from kmodels.base.base_model import load_hf_state_dict
    from kmodels.models.inceptionv4 import InceptionV4
    from kmodels.models.inceptionv4.config import INCEPTIONV4_CONFIG

    for variant, cfg in INCEPTIONV4_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = load_hf_state_dict(f"timm/{timm_id}")
        keras_model = InceptionV4.from_weights(variant, load_weights=False)
        transfer_inceptionv4_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
