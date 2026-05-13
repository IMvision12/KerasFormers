"""timm InceptionResNetV2 -> Keras weight transfer."""

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


def _base_mappings() -> Dict[str, str]:
    return {
        "_conv": ".conv",
        "_batchnorm": ".bn",
        "_kernel": ".weight",
        "_gamma": ".weight",
        "_beta": ".bias",
        "_bias": ".bias",
        "_moving_mean": ".running_mean",
        "_moving_variance": ".running_var",
        "mixed_5b_": "mixed_5b.",
        "mixed_6a_": "mixed_6a.",
        "mixed_7a_": "mixed_7a.",
        "repeats_1_": "repeat_1.",
        "repeats_2_": "repeat_2.",
        "branch1_0": "branch1.0",
        "branch1_1": "branch1.1",
        "branch1_2": "branch1.2",
        "branch2_0": "branch2.0",
        "branch2_1": "branch2.1",
        "branch2_2": "branch2.2",
        "branch3_1": "branch3.1",
        "branch0_0": "branch0.0",
        "branch0_1": "branch0.1",
        "block8_": "block8.",
        "predictions": "classif",
    }


def _generate_repeat_mappings() -> Dict[str, str]:
    mappings: Dict[str, str] = {}

    for i in range(10):
        mappings[f"repeat_{i}_"] = f"repeat.{i}."
        mappings[f"repeat_{i}"] = f"repeat.{i}"

    for i in range(20):
        base = f"repeat.1.{i}"
        keras_base = f"repeat_1.{i}"
        mappings[f"{base}_branch1.0"] = f"{keras_base}.branch1.0"
        mappings[f"{base}_branch1.1"] = f"{keras_base}.branch1.1"
        mappings[f"{base}_branch1.2"] = f"{keras_base}.branch1.2"
        mappings[f"{base}_branch0"] = f"{keras_base}.branch0"
        mappings[f"{base}.conv2d"] = f"{keras_base}.conv2d"

    for i in range(9):
        base = f"repeat.2.{i}"
        keras_base = f"repeat_2.{i}"
        mappings[f"{base}_branch1.0"] = f"{keras_base}.branch1.0"
        mappings[f"{base}_branch1.1"] = f"{keras_base}.branch1.1"
        mappings[f"{base}_branch1.2"] = f"{keras_base}.branch1.2"
        mappings[f"{base}_branch0"] = f"{keras_base}.branch0"
        mappings[f"{base}.conv2d"] = f"{keras_base}.conv2d"

    return mappings


WEIGHT_NAME_MAPPING: Dict[str, str] = {
    **_base_mappings(),
    **_generate_repeat_mappings(),
}


def transfer_inception_resnet_v2_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer a timm InceptionResNetV2 state-dict into a Keras :class:`InceptionResNetV2`."""
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
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
        transfer_weights(keras_weight_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import gc

    import keras

    from kmodels.base.base_model import download_hf_state_dict
    from kmodels.models.inception_resnetv2 import InceptionResNetV2Classify
    from kmodels.models.inception_resnetv2.config import INCEPTION_RESNET_V2_CONFIG

    for variant, cfg in INCEPTION_RESNET_V2_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = InceptionResNetV2Classify.from_weights(
            variant, load_weights=False
        )
        transfer_inception_resnet_v2_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
