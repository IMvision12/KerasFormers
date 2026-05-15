"""timm EfficientFormer -> Keras weight transfer."""

import re
from typing import Dict

import numpy as np

from kmodels.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kmodels.weight_utils.weight_split_torch_and_keras import split_model_weights
from kmodels.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_weights,
)

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "stem.conv1": "stem.conv1",
    "stem.norm1": "stem.norm1",
    "stem.conv2": "stem.conv2",
    "stem.norm2": "stem.norm2",
    "downsample.conv": "downsample.conv",
    "downsample.norm": "downsample.norm",
    "pool.pool": "token_mixer.pool",
    "mlp.conv.1": "mlp.fc1",
    "mlp.norm.1": "mlp.norm1",
    "mlp.conv.2": "mlp.fc2",
    "mlp.norm.2": "mlp.norm2",
    "mlp.dense.1": "mlp.fc1",
    "mlp.dense.2": "mlp.fc2",
    "attn.qkv": "token_mixer.qkv",
    "attn.proj": "token_mixer.proj",
    "norm1": "norm1",
    "norm2": "norm2",
    "final.norm": "norm",
    "kernel": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "head.": "head.",
    "head.dist.": "head_dist.",
}


# Keras emits conv/attn block indices densely (0..depth-1), but timm
# interleaves conv (Meta4D) and vit (Meta1D) under a single index space,
# leaving gaps where the conv→vit transition happens. These maps re-index
# Keras' contiguous block ids into the sparse timm ids for each variant.
#
# L1: last stage has depths=4, num_vit=1 -> conv blocks 0,1,2 then vit block 3
#     timm skips index 3 for conv -> vit block at index 4
# L3: last stage has depths=6, num_vit=4 -> conv blocks 0,1 then vit blocks 2,3,4,5
#     timm skips index 2 for first conv->vit transition, so blocks 2..5 map to 3..6
# L7: last stage has depths=8, num_vit=8 -> all vit blocks 0..7
#     no conv blocks in last stage, no skip needed
def _build_block_index_remap(last_stage_depth: int, num_vit: int) -> Dict[str, str]:
    """Compute keras block-id -> timm block-id rename for the last stage.

    Keras numbers all blocks in stage 3 contiguously ``0..depth-1`` (conv
    blocks first, vit blocks last). timm leaves a one-slot gap at the
    conv→vit transition, so vit blocks live at indices
    ``depth - num_vit + 1 .. depth`` (or ``1 .. depth`` when all blocks
    are vit). The remap shifts every keras vit-block index up by 1.

    Iteration order matters: applying the renames in descending source
    index avoids cascading replacements (e.g. ``blocks.5 -> blocks.6``
    would later be hit by ``blocks.6 -> blocks.7``).
    """
    if num_vit == 0:
        return {}
    first_vit_keras = max(0, last_stage_depth - num_vit)
    remap: Dict[str, str] = {}
    for keras_idx in range(last_stage_depth - 1, first_vit_keras - 1, -1):
        remap[f"stages.3.blocks.{keras_idx}"] = f"stages.3.blocks.{keras_idx + 1}"
    return remap


def transfer_efficientformer_weights(
    keras_model,
    state_dict: Dict[str, np.ndarray],
) -> None:
    """Transfer a timm EfficientFormer state-dict into a Keras :class:`EfficientFormer`.

    Args:
        keras_model: A built :class:`EfficientFormer` instance.
        state_dict: Mapping of timm weight names to numpy arrays.
    """
    block_remap = _build_block_index_remap(
        last_stage_depth=keras_model.depths[-1],
        num_vit=keras_model.num_vit,
    )
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable + non_trainable:
        torch_weight_name = keras_weight_name
        torch_weight_name = re.sub(r"_variable(_\d+)?$", "_gamma", torch_weight_name)

        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if ".gamma" in torch_weight_name and ".ls" not in torch_weight_name:
            torch_weight_name = torch_weight_name.replace(".gamma", ".weight")

        for keras_block, torch_block in block_remap.items():
            if keras_block in torch_weight_name:
                torch_weight_name = torch_weight_name.replace(keras_block, torch_block)
                break

        if "attn.attention.biases" in torch_weight_name:
            torch_weight_name = torch_weight_name.replace(
                ".attn.attention.biases", ".token_mixer.attention_biases"
            )

        if "attention_bias_idxs" in torch_weight_name:
            continue

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]

        if "attention_biases" in keras_weight_name:
            keras_weight.assign(
                torch_weight.numpy() if hasattr(torch_weight, "numpy") else torch_weight
            )
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

    from kmodels.base.base_model import download_hf_state_dict
    from kmodels.models.efficientformer import EfficientFormerClassify
    from kmodels.models.efficientformer.config import EFFICIENTFORMER_MODEL_CONFIG

    for variant, cfg in EFFICIENTFORMER_MODEL_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = EfficientFormerClassify.from_weights(variant, load_weights=False)
        transfer_efficientformer_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
