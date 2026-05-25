import gc
from typing import Dict

import keras
import numpy as np

from kerasformers.base.base_model import download_hf_state_dict
from kerasformers.models.nextvit import NextViTImageClassify as NextViT
from kerasformers.models.nextvit.config import NEXTVIT_WEIGHT_CONFIG
from kerasformers.weight_utils import verify_cls_model_equivalence
from kerasformers.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kerasformers.weight_utils.weight_split_torch_and_keras import split_model_weights
from kerasformers.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "e.mhsa": "e_mhsa",
    "group.conv3x3": "group_conv3x3",
    "patch.embed": "patch_embed",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

_E_MHSA_NAME_REPLACEMENTS: Dict[str, str] = {"e.mhsa": "e_mhsa"}


def transfer_nextvit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "e_mhsa" in keras_weight_name:
            transfer_attention_weights(
                keras_weight_name,
                keras_weight,
                state_dict,
                name_replacements=_E_MHSA_NAME_REPLACEMENTS,
            )
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
        transfer_name = keras_weight_name
        if len(keras_weight.shape) == 4 and "conv" not in keras_weight_name.lower():
            transfer_name = "conv_" + keras_weight_name
        transfer_weights(transfer_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import timm

    for variant, meta in NEXTVIT_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = NextViT.from_weights(
            variant, load_weights=False, include_normalization=False
        )
        transfer_nextvit_weights(keras_model, state)

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
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
            raise ValueError(
                "Model equivalence test failed - model outputs do not match for standard input"
            )

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
