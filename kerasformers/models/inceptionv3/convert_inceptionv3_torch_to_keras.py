"""timm InceptionV3 -> Keras weight transfer."""

import re
from typing import Dict

import numpy as np

from kerasformers.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kerasformers.weight_utils.weight_split_torch_and_keras import split_model_weights
from kerasformers.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_conv2d_kernel": ".conv.weight",
    "_batchnorm_gamma": ".bn.weight",
    "_batchnorm_beta": ".bn.bias",
    "_batchnorm_moving_mean": ".bn.running_mean",
    "_batchnorm_moving_variance": ".bn.running_var",
    "classifier_kernel": "fc.weight",
    "classifier_bias": "fc.bias",
}


def _convert_mixed_block_names(name: str) -> str:
    """Convert Mixed block names: ``Mixed_5b_branch1x1`` -> ``Mixed_5b.branch1x1``."""
    pattern = r"(Mixed_[0-9][a-e])_(.+)"
    match = re.match(pattern, name)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return name


def transfer_inceptionv3_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer a timm InceptionV3 state-dict into a Keras :class:`InceptionV3`."""
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)
        torch_weight_name = _convert_mixed_block_names(torch_weight_name)

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
    import timm

    from kerasformers.base.base_model import download_hf_state_dict
    from kerasformers.models.inceptionv3 import InceptionV3Classify
    from kerasformers.models.inceptionv3.config import INCEPTIONV3_WEIGHT_CONFIG
    from kerasformers.weight_utils import verify_cls_model_equivalence

    for variant, meta in INCEPTIONV3_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = InceptionV3Classify.from_weights(variant, load_weights=False)
        transfer_inceptionv3_weights(keras_model, state)

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="torch_to_keras",
            run_performance=False,
        )

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
