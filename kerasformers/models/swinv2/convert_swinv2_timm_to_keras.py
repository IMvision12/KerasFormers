import gc
from typing import Dict

import keras
import numpy as np
from tqdm import tqdm

from kerasformers.base.base_model import download_hf_state_dict
from kerasformers.models.swinv2 import SwinV2ImageClassify
from kerasformers.models.swinv2.config import SWINV2_WEIGHT_CONFIG
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
    "moving_variance": "MOVVAR",
    "moving_mean": "MOVMEAN",
    "_": ".",
    "MOVVAR": "running_var",
    "MOVMEAN": "running_mean",
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
    "predictions": "head.fc",
}

_ATTN_REPLACEMENT: Dict[str, str] = {
    "cpb.mlp": "cpb_mlp",
}

_DIRECT_ATTN_WEIGHTS: Dict[str, str] = {
    "attn.logit.scale": "attn.logit_scale",
    "attn.q.bias": "attn.q_bias",
    "attn.v.bias": "attn.v_bias",
}

_SKIP_DIRECT_ATTN: tuple = (
    "attn.relative.coords.table",
    "attn.relative.position.index",
)


def transfer_swinv2_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        path_parts = keras_weight.path.split("/")

        if len(path_parts) == 2:
            flat = path_parts[-1].replace("_", ".")
            if any(skip in flat for skip in _SKIP_DIRECT_ATTN):
                continue
            matched = next((k for k in _DIRECT_ATTN_WEIGHTS if k in flat), None)
            if matched is not None:
                torch_name = flat.replace(matched, _DIRECT_ATTN_WEIGHTS[matched])
                if torch_name not in state_dict:
                    raise WeightMappingError(keras_weight_name, torch_name)
                value = np.asarray(state_dict[torch_name])
                if value.shape != tuple(keras_weight.shape):
                    value = value.reshape(tuple(keras_weight.shape))
                keras_weight.assign(value)
                continue

        if len(path_parts) >= 3 and "_attn_" in path_parts[-2]:
            transfer_attention_weights(
                keras_weight_name, keras_weight, state_dict, _ATTN_REPLACEMENT
            )
            continue

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
    import timm

    for variant, meta in SWINV2_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = SwinV2ImageClassify.from_weights(
            variant, load_weights=False, include_normalization=False
        )
        transfer_swinv2_weights(keras_model, state)

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
