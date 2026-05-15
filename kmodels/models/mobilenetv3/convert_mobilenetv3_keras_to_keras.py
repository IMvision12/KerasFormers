"""timm MobileNetV3 -> Keras weight transfer.

Exposes :func:`transfer_mobilenetv3_weights` for both the offline
conversion ``__main__`` block (timm checkpoints -> kmodels release
files) and the runtime ``MobileNetV3.from_weights("timm:...")`` path.

Note: the original script for this family was a keras-to-keras
roundtrip from ``keras.applications.MobileNetV3``; this rewrite shifts
to timm as the upstream so the same ``from_timm`` pathway works as the
rest of the kmodels families.
"""

import re
from typing import Dict, List, Tuple

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

# timm mobilenetv3 groups inverted-residual blocks into 6 (large) / 5 (small)
# stages. The keras model uses a flat ``ir_block_{N}`` naming, so we precompute
# (stage, block_in_stage) for each flat index.
_LARGE_STAGES: List[int] = [1, 2, 3, 4, 2, 3]
_SMALL_STAGES: List[int] = [1, 2, 3, 2, 3]


def _flat_to_stage_map(stage_sizes: List[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for stage_idx, count in enumerate(stage_sizes):
        for b in range(count):
            out.append((stage_idx, b))
    return out


_LARGE_FLAT_TO_STAGE = _flat_to_stage_map(_LARGE_STAGES)
_SMALL_FLAT_TO_STAGE = _flat_to_stage_map(_SMALL_STAGES)

# Block 0 in timm uses a depthwise-separable head (no expansion conv): the
# first 1x1 is named ``conv_dw`` only; ``bn1`` after dw, ``conv_pw`` then
# ``bn2``. The keras block also skips expansion when ``expansion_ratio==1``.
_BLOCK_00_KW: Dict[str, str] = {
    "blocks.0.0.batchnorm.2": "blocks.0.0.bn1",
    "blocks.0.0.batchnorm.3": "blocks.0.0.bn2",
    "blocks.0.0.conv.pwl": "blocks.0.0.conv_pw",
}

_BASE_MAPPINGS: Dict[str, str] = {
    "_": ".",
    "stem.conv": "conv_stem",
    "stem.batchnorm": "bn1",
    "final.conv": "conv_head",
    "final.batchnorm": "bn2",
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


def _ir_block_to_timm(name: str, flat_to_stage: List[Tuple[int, int]]) -> str:
    """Translate ``ir_block_{flat_idx}_<rest>`` -> ``blocks.{s}.{b}.<rest>``."""

    def _sub(m):
        flat_idx = int(m.group(1))
        s, b = flat_to_stage[flat_idx]
        return f"blocks.{s}.{b}."

    return re.sub(r"ir.block.(\d+).", _sub, name)


def transfer_mobilenetv3_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer a timm MobileNetV3 state-dict into a Keras :class:`MobileNetV3`.

    Args:
        keras_model: A built :class:`MobileNetV3` instance.
        state_dict: Mapping of timm weight names to numpy arrays.
    """
    flat_to_stage = (
        _LARGE_FLAT_TO_STAGE
        if getattr(keras_model, "config", "large") == "large"
        else _SMALL_FLAT_TO_STAGE
    )

    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        # Convert ir_block_<N>_ -> blocks.<s>.<b>. first
        torch_weight_name = _ir_block_to_timm(torch_weight_name, flat_to_stage)
        for old, new in {**_BLOCK_00_KW, **_BASE_MAPPINGS}.items():
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
    from kmodels.models.mobilenetv3 import MobileNetV3Classify
    from kmodels.models.mobilenetv3.config import MOBILENETV3_MODEL_CONFIG

    for variant, cfg in MOBILENETV3_MODEL_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = MobileNetV3Classify.from_weights(variant, load_weights=False)
        transfer_mobilenetv3_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
