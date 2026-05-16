"""timm ViT -> Keras weight transfer."""

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
    transfer_attention_weights,
    transfer_weights,
)

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "conv1": "patch_embed.proj",
    "pos.embed.pos.embed": "pos_embed",
    "cls.token.cls.token": "cls_token",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "final.layernorm": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head",
}


def transfer_vit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    """Transfer a timm ViT state-dict into a Keras :class:`ViT`."""
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)
        torch_weight_name = re.sub(
            r"pos_embed_variable_\d+$", "pos_embed", torch_weight_name
        )
        torch_weight_name = re.sub(
            r"cls_token_variable_\d+$", "cls_token", torch_weight_name
        )

        if "attention" in torch_weight_name:
            transfer_attention_weights(keras_weight_name, keras_weight, state_dict)
            continue

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]

        if torch_weight_name == "cls_token":
            keras_weight.assign(torch_weight)
            continue

        if torch_weight_name == "pos_embed":
            if torch_weight.shape[1] == keras_weight.shape[1] + 1:
                torch_weight = torch_weight[:, 1:, :]
            keras_weight.assign(torch_weight)
            continue

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
    from kerasformers.models.vit import ViTClassify
    from kerasformers.models.vit.config import VIT_WEIGHT_CONFIG

    for variant, meta in VIT_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ViTClassify.from_weights(variant, load_weights=False)
        transfer_vit_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
