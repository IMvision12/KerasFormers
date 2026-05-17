import gc
import re
from typing import Dict

import keras
import numpy as np
import timm

from kerasformers.base.base_model import download_hf_state_dict
from kerasformers.models.mobilenetv3 import MobileNetV3ImageClassify
from kerasformers.models.mobilenetv3.config import MOBILENETV3_WEIGHT_CONFIG
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

_LARGE_STAGES = [1, 2, 3, 4, 2, 3]
_SMALL_STAGES = [1, 2, 3, 2, 3]

_BLOCK_00 = {
    "blocks.0.0.batchnorm.2": "blocks.0.0.bn1",
    "blocks.0.0.batchnorm.3": "blocks.0.0.bn2",
    "blocks.0.0.conv.pwl": "blocks.0.0.conv_pw",
}

_BASE_MAPPINGS = {
    "stem.conv": "conv_stem",
    "stem.batchnorm": "bn1",
    "head.conv": "conv_head",
    "se.conv.1": "se.conv_reduce",
    "se.conv.2": "se.conv_expand",
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


def transfer_mobilenetv3_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    stages = (
        _LARGE_STAGES
        if getattr(keras_model, "config", "large") == "large"
        else _SMALL_STAGES
    )
    flat_to_stage = [(s, b) for s, n in enumerate(stages) for b in range(n)]
    final_stage = len(stages)
    final_mapping = {
        "final.conv": f"blocks.{final_stage}.0.conv",
        "final.batchnorm": f"blocks.{final_stage}.0.bn1",
    }
    mapping = {**_BLOCK_00, **final_mapping, **_BASE_MAPPINGS}

    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = re.sub(
            r"ir_block_(\d+)_",
            lambda m: (
                f"blocks.{flat_to_stage[int(m.group(1))][0]}.{flat_to_stage[int(m.group(1))][1]}."
            ),
            keras_weight_name,
        ).replace("_", ".")
        for old, new in mapping.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]
        if (
            keras_weight.ndim == 2
            and torch_weight.ndim == 4
            and torch_weight.shape[-2:] == (1, 1)
        ):
            torch_weight = torch_weight.squeeze(axis=(-1, -2))
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
    for variant, meta in MOBILENETV3_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = MobileNetV3ImageClassify.from_weights(
            variant, load_weights=False, include_normalization=False
        )
        transfer_mobilenetv3_weights(keras_model, state)

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        image_size = keras_model.image_size
        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=(image_size, image_size, 3),
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
