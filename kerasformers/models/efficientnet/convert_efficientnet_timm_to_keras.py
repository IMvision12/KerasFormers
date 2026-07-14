import gc
import re
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
    transfer_weights,
)
from kerasformers.models.efficientnet import EfficientNetImageClassify
from kerasformers.models.efficientnet.efficientnet_config import (
    EFFICIENTNET_WEIGHTS_URLS,
)

_BLOCK_MAPPINGS = {}
for i in range(6):
    block_prefix = f"blocks.0.{i}"
    _BLOCK_MAPPINGS[f"{block_prefix}.conv_pwl"] = f"{block_prefix}.conv_pw"
    _BLOCK_MAPPINGS[f"{block_prefix}.bn2"] = f"{block_prefix}.bn1"
    _BLOCK_MAPPINGS[f"{block_prefix}.bn3"] = f"{block_prefix}.bn2"

_BASE_MAPPINGS = {
    "_kernel": ".weight",
    "_gamma": ".weight",
    "_beta": ".bias",
    "_bias": ".bias",
    "_moving_mean": ".running_mean",
    "_moving_variance": ".running_var",
    "se_": "se.",
    "batchnorm_1": "bn1",
    "batchnorm_2": "bn2",
    "batchnorm_3": "bn3",
    "conv2d_1": "conv_pw",
    "dwconv2d": "conv_dw",
    "conv2d_2": "conv_pwl",
    "predictions": "classifier",
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {**_BASE_MAPPINGS, **_BLOCK_MAPPINGS}


def transfer_efficientnet_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = re.sub(
                r"blocks_(\d+)_(\d+)_",
                lambda m: f"blocks.{m.group(1)}.{m.group(2)}.",
                torch_weight_name,
            )
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

    for variant, meta in EFFICIENTNET_WEIGHTS_URLS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = EfficientNetImageClassify.from_weights(
            variant, load_weights=False, include_normalization=False
        )
        transfer_efficientnet_weights(keras_model, state)

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
