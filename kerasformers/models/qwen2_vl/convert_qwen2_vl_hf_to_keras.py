"""On-the-fly weight conversion for Qwen2-VL (HF safetensors -> Keras).

Follows the library's name-mapped convention (see CLIP / DINOv3 / DETR): driven
off the Keras model's own weights, each weight's hierarchical ``path`` is mapped
to the HF tensor name and assigned via the shared ``transfer_weights`` helper.
The one tensor needing manual handling is the vision Conv3d patch embed
(``(embed_dim, in*t*p*p)`` reshaped to feed the Keras ``Dense``).

Keys use the flat legacy layout: text under ``model.*``, vision under
``visual.*``, tied LM head (no ``lm_head.weight``) for the 2B. The
``__main__`` block runs a local logit-parity check against the HF reference.
"""

import numpy as np

from kerasformers.weight_utils.custom_exception import WeightMappingError
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights

from .qwen2_vl_model import Qwen2VLModel


def normalize_state(state):
    """Accept either the legacy (``visual.*`` / ``model.*``) hub layout or a
    recent ``state_dict()`` (``model.visual.*`` / ``model.language_model.*``)."""
    out = {}
    for k, v in state.items():
        if k.startswith("model.visual."):
            k = k[len("model.") :]
        elif k.startswith("model.language_model."):
            k = "model." + k[len("model.language_model.") :]
        out[k] = v
    return out


def hf_weight_name(path):
    """Map a Keras weight ``path`` to its HuggingFace (legacy-layout) name."""
    rest = path.split("/", 1)[1]  # drop the model-name root
    if rest.startswith("visual/"):
        name = rest.replace("/", ".").replace("blocks_", "blocks.")
        name = name.replace(".gamma", ".weight").replace(".beta", ".bias")
        name = name.replace("merger.mlp_fc1", "merger.mlp.0")
        name = name.replace("merger.mlp_fc2", "merger.mlp.2")
        return name.replace(".kernel", ".weight")
    if rest.startswith("embed_tokens/"):
        return "model.embed_tokens.weight"
    if rest.startswith("lm_head"):
        return "lm_head.weight"
    name = (
        rest[len("language_model/") :].replace("/", ".").replace("layers_", "layers.")
    )
    return "model." + name.replace(".kernel", ".weight")


def build_model(model):
    """Materialize all weights with a minimal, self-consistent dummy forward."""
    m = model.spatial_merge_size
    h = w = 2 * m
    grid = np.array([[1, h, w]], dtype=np.int64)
    n_patches = h * w
    n_merged = n_patches // (m * m)
    model(
        {
            "input_ids": np.array(
                [[0] + [model.image_token_id] * n_merged + [1]], dtype="int64"
            ),
            "pixel_values": np.zeros((n_patches, model.patch_dim), dtype="float32"),
            "image_grid_thw": grid,
        }
    )


def transfer_qwen2_vl_weights(keras_model, hf_state_dict):
    """Assign HF Qwen2-VL weights into ``keras_model`` (built if needed)."""
    if not keras_model.built or not keras_model.weights:
        build_model(keras_model)
    state = normalize_state(hf_state_dict)
    for weight in keras_model.weights:
        name = hf_weight_name(weight.path)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        torch_weight = state[name]
        if "patch_embed" in weight.path:
            # Conv3d (embed_dim, in, t, p, p) -> Dense (in*t*p*p, embed_dim)
            torch_weight = np.asarray(torch_weight).reshape(
                np.asarray(torch_weight).shape[0], -1
            )
        transfer_weights(weight.path, weight, torch_weight)


if __name__ == "__main__":
    import gc

    import torch
    from keras import ops
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    HF_ID = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"[1/4] Loading HF {HF_ID} (float32, cpu)")
    hf = Qwen2VLForConditionalGeneration.from_pretrained(
        HF_ID, torch_dtype=torch.float32
    ).eval()
    processor = AutoProcessor.from_pretrained(HF_ID)

    print("[2/4] Building Keras model + transferring weights")
    state = {k: v.detach().cpu().numpy() for k, v in hf.state_dict().items()}
    model = Qwen2VLModel.from_weights(
        HF_ID.replace("Qwen/", "").lower(), load_weights=False
    )
    transfer_qwen2_vl_weights(model, state)

    print("[3/4] Building a real image+text input")
    from PIL import Image

    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[img], return_tensors="pt")

    print("[4/4] Comparing logits")
    with torch.no_grad():
        hf_logits = hf(**inputs).logits.float().cpu().numpy()
    k_logits = ops.convert_to_numpy(
        model(
            {
                "input_ids": inputs["input_ids"].cpu().numpy(),
                "pixel_values": inputs["pixel_values"].float().cpu().numpy(),
                "image_grid_thw": inputs["image_grid_thw"].cpu().numpy(),
            }
        )["logits"]
    )
    diff = float(np.max(np.abs(hf_logits - k_logits)))
    print(f"  max abs logit diff: {diff:.6e}")
    assert diff < 1e-2, f"parity too high: {diff:.6e}"
    del hf, state
    gc.collect()
