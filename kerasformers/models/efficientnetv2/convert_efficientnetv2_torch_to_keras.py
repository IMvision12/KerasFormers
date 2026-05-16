"""timm EfficientNetV2 (TF) -> Keras weight transfer."""

import gc
import re
from typing import Dict

import keras
import numpy as np
import timm

from kerasformers.base.base_model import download_hf_state_dict
from kerasformers.models.efficientnetv2 import EfficientNetV2ImageClassify
from kerasformers.models.efficientnetv2.config import EFFICIENTNETV2_WEIGHT_CONFIG
from kerasformers.weight_utils import verify_cls_model_equivalence
from kerasformers.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kerasformers.weight_utils.weight_split_torch_and_keras import split_model_weights
from kerasformers.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

# Block 0 (stage 1) uses a single fused conv that timm names ``conv``/``bn1``
# rather than ``conv_pwl``/``bn2`` like the rest of the FMB blocks. The Keras
# code emits the standard FMB names, so we remap them back to timm's for the
# first stage only.
_BLOCK0_REMAP = {}
for j in range(8):  # XL has the most repeats in stage 0 (4); pad generously.
    prefix = f"blocks.0.{j}"
    _BLOCK0_REMAP[f"{prefix}.conv_pwl"] = f"{prefix}.conv"
    _BLOCK0_REMAP[f"{prefix}.bn2"] = f"{prefix}.bn1"

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_kernel": ".weight",
    "_gamma": ".weight",
    "_beta": ".bias",
    "_bias": ".bias",
    "_moving_mean": ".running_mean",
    "_moving_variance": ".running_var",
    "FMBconv1": "conv_exp",
    "FMBconv2": "conv_pwl",
    "MBconv1": "conv_pw",
    "MBdwconv": "conv_dw",
    "MBconv2": "conv_pwl",
    "batchnorm1": "bn1",
    "batchnorm2": "bn2",
    "batchnorm3": "bn3",
    "se_": "se.",
    "predictions": "classifier",
    **_BLOCK0_REMAP,
}


def transfer_efficientnetv2_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer a timm EfficientNetV2 state-dict into a Keras :class:`EfficientNetV2`."""
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        torch_weight_name = re.sub(
            r"blocks_(\d+)_(\d+)_", r"blocks.\1.\2.", torch_weight_name
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
    for variant, meta in EFFICIENTNETV2_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = EfficientNetV2ImageClassify.from_weights(
            variant, load_weights=False
        )
        transfer_efficientnetv2_weights(keras_model, state)

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="torch_to_keras",
            run_performance=False,
        )
        if not results["standard_input"]:
            raise ValueError(
                "Model equivalence test failed - model outputs do not match for standard input"
            )

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
