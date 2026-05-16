"""timm ResNet -> Keras weight transfer.

Exposes :func:`transfer_resnet_weights` for both the offline conversion
``__main__`` block (timm checkpoints -> kerasformers release files) and the
runtime ``ResNet.from_weights("timm:...")`` path.
"""

import gc
from typing import Dict

import keras
import numpy as np
import timm

from kerasformers.base.base_model import download_hf_state_dict
from kerasformers.models.resnet import ResNetImageClassify
from kerasformers.models.resnet.config import RESNET_WEIGHT_CONFIG
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

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "resnet_layer": "layer",
    "_": ".",
    "downsample.conv": "downsample.0",
    "downsample.batchnorm": "downsample.1",
    "batchnorm1": "bn1",
    "batchnorm2": "bn2",
    "batchnorm3": "bn3",
    "dense1": "fc1",
    "dense2": "fc2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "fc",
}


def transfer_resnet_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    """Transfer a timm ResNet state-dict into a Keras :class:`ResNet`.

    Args:
        keras_model: A built :class:`ResNet` instance.
        state_dict: Mapping of timm weight names to numpy arrays (e.g.
            from ``timm.create_model(...).state_dict()`` with each
            tensor moved to CPU + ``.numpy()``).
    """
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
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
    for variant, meta in RESNET_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ResNetImageClassify.from_weights(variant, load_weights=False)
        transfer_resnet_weights(keras_model, state)

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
