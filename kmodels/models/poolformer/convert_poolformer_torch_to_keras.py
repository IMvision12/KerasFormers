"""timm PoolFormer -> Keras weight transfer."""

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
    "stage": "stages",
    "block": "blocks",
    "conv.1": "fc1",
    "conv.2": "fc2",
    "groupnorm.1": "norm1",
    "groupnorm.2": "norm2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "predictions": "head.fc",
    "layernorm": "head.norm",
}


def transfer_poolformer_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    """Transfer a timm PoolFormer state-dict into a Keras :class:`PoolFormer`."""
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)
        torch_weight_name = re.sub(
            r"layerscale\.(\d+)\.variable(?:\.\d+)?$",
            r"layer_scale\1.scale",
            torch_weight_name,
        )

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
    from kmodels.models.poolformer import PoolFormer
    from kmodels.models.poolformer.config import POOLFORMER_CONFIG

    for variant, cfg in POOLFORMER_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = load_hf_state_dict(f"timm/{timm_id}")
        keras_model = PoolFormer.from_weights(variant, load_weights=False)
        transfer_poolformer_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
