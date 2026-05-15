"""timm MobileNetV2 -> Keras weight transfer.

Exposes :func:`transfer_mobilenetv2_weights` for both the offline
conversion ``__main__`` block (timm checkpoints -> kmodels release
files) and the runtime ``MobileNetV2.from_weights("timm:...")`` path.
"""

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

# Block 0.0 has no expansion, so its batchnorms shift down by one and the
# pointwise-out layer is named conv_pw (no -l suffix) in timm.
_BLOCK_00 = {
    "blocks.0.0.batchnorm.1": "blocks.0.0.bn1",
    "blocks.0.0.batchnorm.3": "blocks.0.0.bn2",
    "blocks.0.0.conv.pwl": "blocks.0.0.conv_pw",
}

_BASE_MAPPINGS = {
    "_": ".",
    "stem.conv": "conv_stem",
    "stem.batchnorm": "bn1",
    "head.conv": "conv_head",
    "head.batchnorm": "bn2",
    "batchnorm.1": "bn1",
    "batchnorm.2": "bn2",
    "batchnorm.3": "bn3",
    "conv.pw": "conv_pw",
    "dwconv": "conv_dw",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "classifier",
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {**_BLOCK_00, **_BASE_MAPPINGS}


def transfer_mobilenetv2_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer a timm MobileNetV2 state-dict into a Keras :class:`MobileNetV2`.

    Args:
        keras_model: A built :class:`MobileNetV2` instance.
        state_dict: Mapping of timm weight names to numpy arrays.
    """
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        torch_weight_name = re.sub(
            r"blocks_(\d+)_(\d+)_",
            lambda m: f"blocks.{m.group(1)}.{m.group(2)}.",
            torch_weight_name,
        )
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
    from kmodels.models.mobilenetv2 import MobileNetV2Classify
    from kmodels.models.mobilenetv2.config import MOBILENETV2_MODEL_CONFIG

    for variant, cfg in MOBILENETV2_MODEL_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = MobileNetV2Classify.from_weights(variant, load_weights=False)
        transfer_mobilenetv2_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
