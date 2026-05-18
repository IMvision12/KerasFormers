import gc
import math
import re
import sys
from typing import Dict

import keras
import numpy as np
import timm

from kerasformers.models.mobilenetv3 import MobileNetV3ImageClassify
from kerasformers.models.mobilenetv3.config import (
    MOBILENETV3_MODEL_CONFIG,
    MOBILENETV3_WEIGHT_CONFIG,
)
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

_BASE_LARGE_STAGES = [1, 2, 3, 4, 2, 3]
_BASE_SMALL_STAGES = [1, 2, 3, 2, 3]


def build_stage0_mapping(num_stage0_blocks: int):
    """Stage 0 in MobileNetV3 contains DS-style blocks (expansion=1) — timm names
    them with ``conv_pw`` / ``bn1`` / ``bn2`` (no expansion conv, so only two
    BNs). Our keras model uses the IR layout (``conv_pwl`` / ``bn2`` / ``bn3``)
    uniformly. Rewrite the stage-0 names for each block in stage 0.
    """
    mapping = {}
    for i in range(num_stage0_blocks):
        mapping[f"blocks.0.{i}.batchnorm.2"] = f"blocks.0.{i}.bn1"
        mapping[f"blocks.0.{i}.batchnorm.3"] = f"blocks.0.{i}.bn2"
        mapping[f"blocks.0.{i}.conv.pwl"] = f"blocks.0.{i}.conv_pw"
    return mapping


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


def stage_counts(config: str, block_count_multiplier: float = 1.0):
    base = _BASE_LARGE_STAGES if config == "large" else _BASE_SMALL_STAGES
    return [math.ceil(n * block_count_multiplier) for n in base]


def build_final_mapping(stage_counts_list, head_count_multiplier: int):
    final_stage = len(stage_counts_list)
    mapping = {}
    if head_count_multiplier == 1:
        mapping["final.batchnorm"] = f"blocks.{final_stage}.0.bn1"
        mapping["final.conv"] = f"blocks.{final_stage}.0.conv"
    else:
        for i in range(head_count_multiplier):
            mapping[f"final.batchnorm.{i}"] = f"blocks.{final_stage}.{i}.bn1"
            mapping[f"final.conv.{i}"] = f"blocks.{final_stage}.{i}.conv"
    return mapping


def transfer_mobilenetv3_weights(
    keras_model,
    state_dict: Dict[str, np.ndarray],
    stage_counts_list,
    head_count_multiplier: int = 1,
) -> None:
    flat_to_stage = [(s, b) for s, n in enumerate(stage_counts_list) for b in range(n)]
    final_mapping = build_final_mapping(stage_counts_list, head_count_multiplier)
    stage0_mapping = build_stage0_mapping(stage_counts_list[0])
    mapping = {**stage0_mapping, **final_mapping, **_BASE_MAPPINGS}

    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = re.sub(
            r"ir_block_(\d+)_",
            lambda m: (
                f"blocks.{flat_to_stage[int(m.group(1))][0]}."
                f"{flat_to_stage[int(m.group(1))][1]}."
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
    sys.setrecursionlimit(10000)

    for variant, meta in MOBILENETV3_WEIGHT_CONFIG.items():
        model_cfg = dict(MOBILENETV3_MODEL_CONFIG[meta["model"]])
        model_cfg.pop("num_classes", None)
        block_count_multiplier = model_cfg.get("block_count_multiplier", 1.0)
        head_count_multiplier = model_cfg.get("head_count_multiplier", 1)
        scounts = stage_counts(model_cfg["config"], block_count_multiplier)

        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        state = {
            k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()
        }
        num_classes = int(state["classifier.weight"].shape[0])

        keras_model = MobileNetV3ImageClassify(
            **model_cfg,
            num_classes=num_classes,
            include_normalization=False,
        )

        transfer_mobilenetv3_weights(keras_model, state, scounts, head_count_multiplier)

        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="torch_to_keras",
            run_performance=False,
            atol=1e-4,
            rtol=1e-4,
        )
        if not results["standard_input"]:
            raise ValueError(f"{variant}: model equivalence test failed")

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
