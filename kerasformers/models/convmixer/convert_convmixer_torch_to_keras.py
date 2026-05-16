"""timm ConvMixerClassify -> Keras weight transfer."""

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
    "_": ".",
    "stem.conv2d": "stem.0",
    "stem.batchnorm": "stem.2",
    "mixer.block.": "blocks.",
    ".depthwise": ".0.fn.0",
    ".batchnorm.1": ".0.fn.2",
    ".add": ".0",
    ".conv2d": ".1",
    ".batchnorm.2": ".3",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head",
}


def transfer_convmixer_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    """Transfer a timm ConvMixerClassify state-dict into a Keras :class:`ConvMixerClassify`."""
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
    import gc

    import keras

    from kerasformers.base.base_model import download_hf_state_dict
    from kerasformers.models.convmixer import ConvMixerClassify
    from kerasformers.models.convmixer.config import CONVMIXER_WEIGHT_CONFIG

    for variant, meta in CONVMIXER_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ConvMixerClassify.from_weights(variant, load_weights=False)
        transfer_convmixer_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
