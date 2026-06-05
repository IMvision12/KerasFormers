import gc
from typing import Dict

import keras
import numpy as np
from tqdm import tqdm

from kerasformers.conversion import verify_cls_model_equivalence
from kerasformers.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kerasformers.conversion.hf_download_utils import download_hf_state_dict
from kerasformers.conversion.weight_split_util import split_model_weights
from kerasformers.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)
from kerasformers.models.swin import SwinImageClassify
from kerasformers.models.swin.config import SWIN_WEIGHT_CONFIG

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "stem.conv": "patch_embed.proj",
    "stem.norm": "patch_embed.norm",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "dense.1": "fc1",
    "dense.2": "fc2",
    "pm.layernorm": "norm",
    "pm.dense": "reduction",
    "final.norm": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head.fc",
}


def transfer_swin_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "relative.position.bias.table" in torch_weight_name:
            layer_name = keras_weight.path.split("/")[-1]
            layer_name = layer_name.replace("_", ".").replace(
                "relative.position.bias.table", "relative_position_bias_table"
            )
            keras_weight.assign(state_dict[layer_name])
            continue

        if "window.attention" in torch_weight_name:
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
    import timm

    for variant, meta in SWIN_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = SwinImageClassify.from_weights(
            variant, load_weights=False, include_normalization=False
        )
        transfer_swin_weights(keras_model, state)

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
