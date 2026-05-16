import gc
import re
from typing import Dict, List, Tuple

import keras
import numpy as np
import torch
from tqdm import tqdm

from kerasformers.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kerasformers.weight_utils.weight_split_torch_and_keras import split_model_weights
from kerasformers.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
    "_": ".",
    "conv1": "embeddings.patch_embeddings.projection",
    "pos.embed.pos.embed": "embeddings.position_embeddings",
    "cls.token.cls.token": "embeddings.cls_token",
    "blocks.": "encoder.layer.",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "final.layernorm": "layernorm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def _resolve_attention_qkv(keras_weight_path: str):
    """Return the three HF Q/K/V keys for a fused Keras qkv weight."""
    layer_segment = keras_weight_path.split("/")[-2]
    m = re.match(r"blocks_(\d+)_attn_qkv$", layer_segment)
    if m is None:
        return None
    idx = int(m.group(1))
    base = f"encoder.layer.{idx}.attention.attention"
    suffix = "weight" if "kernel" in keras_weight_path else "bias"
    return (f"{base}.query.{suffix}", f"{base}.key.{suffix}", f"{base}.value.{suffix}")


def _resolve_attention_proj(keras_weight_path: str):
    """Map ``blocks_{i}_attn_proj`` to HF attention output dense."""
    layer_segment = keras_weight_path.split("/")[-2]
    m = re.match(r"blocks_(\d+)_attn_proj$", layer_segment)
    if m is None:
        return None
    idx = int(m.group(1))
    suffix = "weight" if "kernel" in keras_weight_path else "bias"
    return f"encoder.layer.{idx}.attention.output.dense.{suffix}"


def _resolve_layer_scale(keras_weight_path: str):
    """Map ``blocks_{i}_layerscale_{1,2}/variable*`` to HF lambda1."""
    layer_segment = keras_weight_path.split("/")[-2]
    m = re.match(r"blocks_(\d+)_layerscale_(1|2)$", layer_segment)
    if m is None:
        return None
    idx = int(m.group(1))
    which = m.group(2)
    return f"encoder.layer.{idx}.layer_scale{which}.lambda1"


def _fuse_qkv(state_dict, q_key, k_key, v_key):
    """Concatenate HF Q, K, V weights along output dim to match fused qkv."""
    q = state_dict[q_key]
    k = state_dict[k_key]
    v = state_dict[v_key]
    if hasattr(q, "numpy"):
        q, k, v = q.numpy(), k.numpy(), v.numpy()
    return np.concatenate([q, k, v], axis=0)


def _interpolate_pos_embed(pos_embed, target_num_patches: int):
    """Bilinearly resize a DINOv2 position-embedding tensor."""
    if not isinstance(pos_embed, torch.Tensor):
        pos_embed = torch.from_numpy(np.asarray(pos_embed))
    cls_pe = pos_embed[:, :1]
    spatial_pe = pos_embed[:, 1:]
    src_num = spatial_pe.shape[1]
    src_size = int(round(src_num**0.5))
    tgt_size = int(round(target_num_patches**0.5))
    if src_size == tgt_size:
        return pos_embed.numpy()

    dim = spatial_pe.shape[-1]
    spatial_pe = spatial_pe.reshape(1, src_size, src_size, dim).permute(0, 3, 1, 2)
    spatial_pe = torch.nn.functional.interpolate(
        spatial_pe.float(),
        size=(tgt_size, tgt_size),
        mode="bicubic",
        align_corners=False,
    )
    spatial_pe = spatial_pe.permute(0, 2, 3, 1).reshape(1, tgt_size * tgt_size, dim)
    return torch.cat([cls_pe, spatial_pe], dim=1).numpy()


def _strip_prefix(state_dict, prefix):
    """Strip a common prefix from all state-dict keys that start with it.

    For ``Dinov2ForImageClassification`` and similar task wrappers, the
    backbone keys are nested under ``dinov2.*`` and there is an
    additional ``classifier.*`` head. We strip the prefix from keys
    that have it; other keys (classifier head, etc.) are dropped.
    """
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def transfer_dino_v2_weights(
    keras_model: keras.Model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer DINOv2 weights from a HuggingFace state-dict.

    Handles fused QKV concatenation, attention output projection,
    LayerScale, and bicubic interpolation of the position embeddings
    when the Keras input shape differs from HF's training resolution.

    Also strips a ``dinov2.`` prefix when loading from
    ``Dinov2ForImageClassification`` or other task-head fine-tunes,
    discarding the classifier head.

    Args:
        keras_model: A ``DinoV2Backbone`` instance.
        hf_state_dict: Mapping of HF weight names to numpy arrays from
            ``Dinov2Model.state_dict()`` or any ``Dinov2For*`` variant.
    """
    hf_state_dict = _strip_prefix(hf_state_dict, "dinov2.")
    trainable, non_trainable = split_model_weights(keras_model)
    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring DINOv2 weights"
    ):
        path = keras_weight.path

        if "_attn_qkv" in path:
            qkv_keys = _resolve_attention_qkv(path)
            if qkv_keys is None:
                raise WeightMappingError(keras_weight_name, path)
            for key in qkv_keys:
                if key not in hf_state_dict:
                    raise WeightMappingError(keras_weight_name, key)
            fused = _fuse_qkv(hf_state_dict, *qkv_keys)
            transfer_weights(keras_weight_name, keras_weight, fused)
            continue

        if "_attn_proj" in path:
            hf_key = _resolve_attention_proj(path)
            if hf_key is None or hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, str(hf_key))
            transfer_weights(keras_weight_name, keras_weight, hf_state_dict[hf_key])
            continue

        if "_layerscale_" in path:
            hf_key = _resolve_layer_scale(path)
            if hf_key is None or hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, str(hf_key))
            w = hf_state_dict[hf_key]
            keras_weight.assign(w.numpy() if hasattr(w, "numpy") else w)
            continue

        torch_weight_name = keras_weight_name
        for old, new in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        torch_weight_name = re.sub(
            r"pos_embed_variable_\d+$",
            "embeddings.position_embeddings",
            torch_weight_name,
        )
        torch_weight_name = re.sub(
            r"cls_token_variable_\d+$",
            "embeddings.cls_token",
            torch_weight_name,
        )

        if torch_weight_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = hf_state_dict[torch_weight_name]

        if torch_weight_name == "embeddings.cls_token":
            keras_weight.assign(
                torch_weight.numpy() if hasattr(torch_weight, "numpy") else torch_weight
            )
            continue

        if torch_weight_name == "embeddings.position_embeddings":
            target_num_patches = keras_weight.shape[1] - 1
            resized = _interpolate_pos_embed(torch_weight, target_num_patches)
            keras_weight.assign(resized)
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


DINOV2_CONVERSION_CONFIG: List[Tuple[str, str]] = [
    ("dinov2_vits14", "facebook/dinov2-small"),
    ("dinov2_vitb14", "facebook/dinov2-base"),
    ("dinov2_vitl14", "facebook/dinov2-large"),
]


if __name__ == "__main__":
    from transformers import Dinov2Model

    from kerasformers.models.dino_v2 import DinoV2Backbone

    for variant, hf_id in DINOV2_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = Dinov2Model.from_pretrained(hf_id).eval()
        hf_state_dict = dict(hf_model.state_dict())

        keras_model = DinoV2Backbone.from_weights(
            variant,
            load_weights=False,
            input_shape=(224, 224, 3),
            include_normalization=False,
        )

        transfer_dino_v2_weights(keras_model, hf_state_dict)

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
        with torch.no_grad():
            hf_out = (
                hf_model(pixel_values=torch.from_numpy(x_np))
                .last_hidden_state.cpu()
                .numpy()
            )
        k_in = np.transpose(x_np, (0, 2, 3, 1))
        k_raw = keras_model(k_in, training=False)
        # Backbone output is a list of intermediate features; take the last block's output
        last = k_raw[-1]
        k_out = (
            last.detach().cpu().numpy() if hasattr(last, "detach") else np.asarray(last)
        )
        diff = float(np.abs(k_out - hf_out).max())
        if diff > 1e-3:
            raise ValueError(f"{variant}: max diff {diff:.2e}")
        print(f"  Verification OK (max diff = {diff:.2e})")

        model_filename = f"{variant}.weights.h5"
        keras_model.save_weights(model_filename)
        print(f"  Saved -> {model_filename}")

        del keras_model, hf_model, hf_state_dict
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
