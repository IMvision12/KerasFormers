"""timm MobileViT -> Keras weight transfer.

Exposes :func:`transfer_mobilevit_weights` for both the offline
conversion ``__main__`` block (timm checkpoints -> kmodels release
files) and the runtime ``MobileViT.from_weights("timm:...")`` path.
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
    transfer_attention_weights,
    transfer_weights,
)

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "batchnorm": "bn",
    "ir.conv.1": "conv1_1x1.conv",
    "ir.bn.1": "conv1_1x1.bn",
    "ir.dwconv": "conv2_kxk.conv",
    "ir.bn.2": "conv2_kxk.bn",
    "ir.conv.2": "conv3_1x1.conv",
    "ir.bn.3": "conv3_1x1.bn",
    "mv.conv.1": "conv_kxk.conv",
    "mv.bn.1": "conv_kxk.bn",
    "mv.conv.2": "conv_1x1",
    "layernorm": "norm",
    "norm.1": "norm1",
    "norm.2": "norm2",
    "mv.conv.3": "conv_proj.conv",
    "mv.bn.2": "conv_proj.bn",
    "mv.conv.4": "conv_fusion.conv",
    "mv.bn.3": "conv_fusion.bn",
    "final.conv": "final_conv.conv",
    "final.bn": "final_conv.bn",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head.fc",
}


def transfer_mobilevit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    """Transfer a timm MobileViT state-dict into a Keras :class:`MobileViT`.

    Args:
        keras_model: A built :class:`MobileViT` instance.
        state_dict: Mapping of timm weight names to numpy arrays.
    """
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "attention" in torch_weight_name:
            transfer_attention_weights(keras_weight_name, keras_weight, state_dict)
            continue

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
    from kmodels.models.mobilevit import MobileViTClassify as MobileViT
    from kmodels.models.mobilevit.config import MOBILEVIT_MODEL_CONFIG

    for variant, cfg in MOBILEVIT_MODEL_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = MobileViT.from_weights(variant, load_weights=False)
        transfer_mobilevit_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
