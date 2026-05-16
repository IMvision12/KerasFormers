"""HuggingFace SigLIP -> Keras weight transfer.

Splits the conversion into a callable :func:`transfer_siglip_weights`
that takes a Keras :class:`~kerasformers.models.siglip.SigLIPModel` and an
HF state dict (numpy values), plus a ``__main__`` block that runs the
google -> kerasformers conversion for every variant.
"""

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

WEIGHT_NAME_MAPPING = {
    "_": ".",
    "vision.model": "vision_model",
    "text.model": "text_model",
    "patch.embedding.conv": "patch_embedding",
    "position.embedding.embeddings": "position_embedding.weight",
    "token.embedding.embeddings": "token_embedding.weight",
    "text_model.post_layernorm": "text_model.final_layer_norm",
    "layernorm.1": "layer_norm1",
    "layernorm.2": "layer_norm2",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "vision_model.final.layernorm": "vision_model.post_layernorm",
    "text_model.final.layernorm": "text_model.final_layer_norm",
    "probe.probe": "probe",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
}

ATTN_NAME_REPLACE = {
    "_": ".",
    "self.attn": "self_attn",
    "vision.model": "vision_model",
    "text.model": "text_model",
    "in.proj": "in_proj",
    "out.proj": "out_proj",
    "q.proj": "q_proj",
    "k.proj": "k_proj",
    "v.proj": "v_proj",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
}


def transfer_siglip_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    """Transfer HuggingFace SigLIP / SigLIP-2 weights into a Keras model.

    Args:
        keras_model: A :class:`SigLIPModel` or :class:`SigLIPZeroShotClassify`
            instance.
        hf_state_dict: Mapping of HF weight names to numpy arrays from
            ``SiglipModel.state_dict()``.
    """
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "attention" in torch_weight_name:
            if "in_proj" in keras_weight.path:
                if "kernel" in keras_weight.path:
                    keras_weight.assign(
                        np.transpose(
                            hf_state_dict["vision_model.head.attention.in_proj_weight"]
                        )
                    )
                else:
                    keras_weight.assign(
                        hf_state_dict["vision_model.head.attention.in_proj_bias"]
                    )
                continue
            transfer_attention_weights(
                keras_weight_name, keras_weight, hf_state_dict, ATTN_NAME_REPLACE
            )
            continue

        if "probe" in torch_weight_name:
            keras_weight.assign(hf_state_dict["vision_model.head.probe"])
            continue

        if "logit" in torch_weight_name:
            if torch_weight_name.split(".")[-1] == "scale":
                keras_weight.assign(hf_state_dict["logit_scale"].reshape(()))
            else:
                keras_weight.assign(hf_state_dict["logit_bias"].reshape(()))
            continue

        if "position.ids" in torch_weight_name:
            if "vision_model" in torch_weight_name:
                key = "vision_model.embeddings.position_ids"
            else:
                key = "text_model.embeddings.position_ids"
            if key in hf_state_dict:
                keras_weight.assign(hf_state_dict[key])
            continue

        if "head.attention" in torch_weight_name:
            continue

        if torch_weight_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = hf_state_dict[torch_weight_name]
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


def transfer_siglip_image_classify_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer HuggingFace ``SiglipForImageClassification`` weights.

    Loads the SigLIP vision encoder (no text encoder, no attention
    pooling, no ``logit_scale``/``logit_bias`` — none of those exist in
    the Keras :class:`SigLIPImageClassify` graph) plus the final
    ``classifier`` Dense head. If the source is a base SigLIP checkpoint
    without classifier weights, the head stays randomly initialized.
    """
    has_classifier = (
        "classifier.weight" in hf_state_dict and "classifier.bias" in hf_state_dict
    )
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        if keras_weight_name in ("classifier_kernel", "classifier_bias"):
            if not has_classifier:
                continue
            if "kernel" in keras_weight.path:
                keras_weight.assign(np.transpose(hf_state_dict["classifier.weight"]))
            else:
                keras_weight.assign(hf_state_dict["classifier.bias"])
            continue

        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, hf_state_dict, ATTN_NAME_REPLACE
            )
            continue

        if "position.ids" in torch_weight_name:
            key = "vision_model.embeddings.position_ids"
            if key in hf_state_dict:
                keras_weight.assign(hf_state_dict[key])
            continue

        if torch_weight_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = hf_state_dict[torch_weight_name]
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
    from transformers import SiglipModel

    from kerasformers.models.siglip import SigLIPZeroShotClassify

    SIGLIP_CONVERSION_CONFIG = [
        ("siglip_base_p16_224", "google/siglip-base-patch16-224"),
        ("siglip_base_p16_256", "google/siglip-base-patch16-256"),
        ("siglip_base_p16_384", "google/siglip-base-patch16-384"),
        ("siglip_base_p16_512", "google/siglip-base-patch16-512"),
        ("siglip_large_p16_256", "google/siglip-large-patch16-256"),
        ("siglip_large_p16_384", "google/siglip-large-patch16-384"),
    ]

    for variant, hf_id in SIGLIP_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = SiglipModel.from_pretrained(hf_id).eval()
        state = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = SigLIPZeroShotClassify.from_weights(variant, load_weights=False)
        transfer_siglip_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, hf_model, state
        keras.backend.clear_session()
        gc.collect()
