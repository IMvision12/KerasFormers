import gc
from typing import Dict

import keras
import numpy as np
from transformers import AutoModel

from kerasformers.models.clip import CLIPZeroShotClassify
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

WEIGHT_NAME_MAPPING = {
    "_": ".",
    "vision.model": "vision_model",
    "text.model": "text_model",
    "conv": "embeddings.patch_embedding",
    "class.embedding": "class_embedding",
    "pos.embed": "position_embedding.weight",
    "vision_model.layernorm.1": "vision_model.pre_layrnorm",
    "text_model.encoder": "text_model.encoder.layers",
    "vision_model.encoder": "vision_model.encoder.layers",
    "text_model.layernorm": "text_model.final_layer_norm",
    "layernorm.1": "layer_norm1",
    "layernorm.2": "layer_norm2",
    "vision_model.layer_norm2": "vision_model.post_layernorm",
    "text.projection": "text_projection",
    "visual.projection": "visual_projection",
    "logit_scale_logit_scale": "logit_scale",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
}

ATTN_NAME_REPLACE = {
    "text.model": "text_model",
    "vision.model": "vision_model",
    "encoder": "encoder.layers",
    "attn": "self_attn",
    "q.proj": "q_proj",
    "k.proj": "k_proj",
    "v.proj": "v_proj",
    "out.proj": "out_proj",
}


def _strip_model_prefix(state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not any(k.startswith("model.") for k in state_dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            out[k[len("model.") :]] = v
        else:
            out[k] = v
    return out


def transfer_clip_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    state = _strip_model_prefix(hf_state_dict)
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name: str = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, state, ATTN_NAME_REPLACE
            )
            continue

        if keras_weight_name == "text_model_embedding_embeddings":
            if "token_embedding" in keras_weight.path:
                keras_weight.assign(
                    state["text_model.embeddings.token_embedding.weight"]
                )
                continue
            if "positional_embedding" in keras_weight.path:
                keras_weight.assign(
                    state["text_model.embeddings.position_embedding.weight"]
                )
                continue

        if keras_weight_name == "logit_scale_logit_scale":
            keras_weight.assign(state["logit_scale"])
            continue

        if keras_weight_name == "vision_model_embeddings_pos_embed":
            pos = state["vision_model.embeddings.position_embedding.weight"]
            keras_weight.assign(np.expand_dims(pos, 0))
            continue

        if torch_weight_name not in state:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state[torch_weight_name]
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


def transfer_clip_image_classify_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    state = _strip_model_prefix(hf_state_dict)
    has_classifier = "classifier.weight" in state and "classifier.bias" in state
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        if keras_weight_name in ("classifier_kernel", "classifier_bias"):
            if not has_classifier:
                continue
            if "kernel" in keras_weight.path:
                keras_weight.assign(np.transpose(state["classifier.weight"]))
            else:
                keras_weight.assign(state["classifier.bias"])
            continue

        torch_weight_name: str = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, state, ATTN_NAME_REPLACE
            )
            continue

        if keras_weight_name == "vision_model_embeddings_pos_embed":
            pos = state["vision_model.embeddings.position_embedding.weight"]
            keras_weight.assign(np.expand_dims(pos, 0))
            continue

        if torch_weight_name not in state:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state[torch_weight_name]
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
    CLIP_CONVERSION_CONFIG = [
        ("clip_vit_base_16", "openai/clip-vit-base-patch16"),
        ("clip_vit_base_32", "openai/clip-vit-base-patch32"),
        ("clip_vit_large_14", "openai/clip-vit-large-patch14"),
        ("clip_vit_large_14_336", "openai/clip-vit-large-patch14-336"),
    ]

    for variant, hf_id in CLIP_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = AutoModel.from_pretrained(hf_id).eval()
        state = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = CLIPZeroShotClassify.from_weights(variant, load_weights=False)
        transfer_clip_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, hf_model, state
        keras.backend.clear_session()
        gc.collect()
