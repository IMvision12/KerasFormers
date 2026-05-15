"""timm MobileViTV2 -> Keras weight transfer.

Exposes :func:`transfer_mobilevitv2_weights` for both the offline
conversion ``__main__`` block (timm checkpoints -> kmodels release
files) and the runtime ``MobileViTV2.from_weights("timm:...")`` path.
"""

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
    "stem.batchnorm": "stem.bn",
    "ir.conv.1": "conv1_1x1.conv",
    "ir.batchnorm.1": "conv1_1x1.bn",
    "ir.dwconv": "conv2_kxk.conv",
    "ir.batchnorm.2": "conv2_kxk.bn",
    "ir.conv.2": "conv3_1x1.conv",
    "ir.batchnorm.3": "conv3_1x1.bn",
    "mv2.dwconv": "conv_kxk.conv",
    "mv2.batchnorm.1": "conv_kxk.bn",
    "mv2.conv.1": "conv_1x1",
    "groupnorm.1": "norm1",
    "attn.conv.1": "attn.qkv_proj",
    "attn.conv.2": "attn.out_proj",
    "groupnorm.2": "norm2",
    "mlp.conv.1": "mlp.fc1",
    "mlp.conv.2": "mlp.fc2",
    "groupnorm": "norm",
    "mv2.proj.conv": "conv_proj.conv",
    "mv2.proj.batchnorm": "conv_proj.bn",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head.fc",
}


def transfer_mobilevitv2_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer a timm MobileViTV2 state-dict into a Keras :class:`MobileViTV2`.

    Args:
        keras_model: A built :class:`MobileViTV2` instance.
        state_dict: Mapping of timm weight names to numpy arrays.
    """
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
    from kmodels.models.mobilevitv2 import MobileViTV2Classify as MobileViTV2
    from kmodels.models.mobilevitv2.config import MOBILEVITV2_WEIGHT_CONFIG

    for variant, meta in MOBILEVITV2_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = MobileViTV2.from_weights(variant, load_weights=False)
        transfer_mobilevitv2_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
