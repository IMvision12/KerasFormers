"""timm EfficientNet (TF) -> Keras weight transfer."""

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

_BLOCK_MAPPINGS = {}
for i in range(6):
    block_prefix = f"blocks.0.{i}"
    _BLOCK_MAPPINGS[f"{block_prefix}.conv_pwl"] = f"{block_prefix}.conv_pw"
    _BLOCK_MAPPINGS[f"{block_prefix}.bn2"] = f"{block_prefix}.bn1"
    _BLOCK_MAPPINGS[f"{block_prefix}.bn3"] = f"{block_prefix}.bn2"

_BASE_MAPPINGS = {
    "_kernel": ".weight",
    "_gamma": ".weight",
    "_beta": ".bias",
    "_bias": ".bias",
    "_moving_mean": ".running_mean",
    "_moving_variance": ".running_var",
    "se_": "se.",
    "batchnorm_1": "bn1",
    "batchnorm_2": "bn2",
    "batchnorm_3": "bn3",
    "conv2d_1": "conv_pw",
    "dwconv2d": "conv_dw",
    "conv2d_2": "conv_pwl",
    "predictions": "classifier",
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {**_BASE_MAPPINGS, **_BLOCK_MAPPINGS}


def transfer_efficientnet_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer a timm EfficientNet state-dict into a Keras :class:`EfficientNet`."""
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = re.sub(
                r"blocks_(\d+)_(\d+)_",
                lambda m: f"blocks.{m.group(1)}.{m.group(2)}.",
                torch_weight_name,
            )
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
        transfer_weights(keras_weight_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import gc

    import keras

    from kmodels.base.base_model import load_hf_state_dict
    from kmodels.models.efficientnet import EfficientNet
    from kmodels.models.efficientnet.config import EFFICIENTNET_CONFIG

    for variant, cfg in EFFICIENTNET_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = load_hf_state_dict(f"timm/{timm_id}")
        keras_model = EfficientNet.from_weights(variant, load_weights=False)
        transfer_efficientnet_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
