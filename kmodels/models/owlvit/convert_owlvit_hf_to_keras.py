import os
from typing import Any, Dict, List

import keras
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from kmodels.models import owlvit

model_configs: List[Dict[str, Any]] = [
    {
        "keras_model_cls": owlvit.OwlViTBasePatch32,
        "hf_model_name": "google/owlvit-base-patch32",
        "image_size": 768,
    },
    {
        "keras_model_cls": owlvit.OwlViTBasePatch16,
        "hf_model_name": "google/owlvit-base-patch16",
        "image_size": 768,
    },
    {
        "keras_model_cls": owlvit.OwlViTLargePatch14,
        "hf_model_name": "google/owlvit-large-patch14",
        "image_size": 840,
    },
]


def _walk(parent: keras.layers.Layer, path: str) -> keras.layers.Layer:
    parts = path.split("/")
    if isinstance(parent, keras.Model):
        cur: Any = parent.get_layer(parts[0])
        parts = parts[1:]
    else:
        cur = parent
    for part in parts:
        nxt = getattr(cur, part, None)
        if nxt is None:
            for child in getattr(cur, "encoder_layers", []) or []:
                if getattr(child, "name", None) == part:
                    nxt = child
                    break
        if nxt is None:
            raise AttributeError(f"Could not resolve '{part}' on path '{path}'.")
        cur = nxt
    return cur


def _load_state_dict(hf_repo: str) -> Dict[str, np.ndarray]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    token = os.environ.get("HF_TOKEN")
    try:
        path = hf_hub_download(hf_repo, "model.safetensors", token=token)
        from safetensors.torch import safe_open

        sd: Dict[str, np.ndarray] = {}
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                sd[k] = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
        return sd
    except RemoteEntryNotFoundError:
        path = hf_hub_download(hf_repo, "pytorch_model.bin", token=token)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        return {k: v.cpu().numpy() for k, v in raw.items()}


def _transfer_encoder(
    keras_model: keras.Model,
    pytorch_state_dict: Dict[str, np.ndarray],
    hf_prefix: str,
    keras_prefix: str,
    num_layers: int,
    desc: str,
) -> None:
    for i in tqdm(range(num_layers), desc=desc):
        h = f"{hf_prefix}.encoder.layers.{i}"
        k = f"{keras_prefix}_layers_{i}"
        self_attn = _walk(keras_model, f"{k}_self_attn")
        for proj in ("k_proj", "v_proj", "q_proj", "out_proj"):
            dense = getattr(self_attn, proj)
            dense.kernel.assign(pytorch_state_dict[f"{h}.self_attn.{proj}.weight"].T)
            dense.bias.assign(pytorch_state_dict[f"{h}.self_attn.{proj}.bias"])
        for ln in ("layer_norm1", "layer_norm2"):
            layer = _walk(keras_model, f"{k}_{ln}")
            layer.weights[0].assign(pytorch_state_dict[f"{h}.{ln}.weight"])
            layer.weights[1].assign(pytorch_state_dict[f"{h}.{ln}.bias"])
        for fc in ("fc1", "fc2"):
            layer = _walk(keras_model, f"{k}_mlp_{fc}")
            layer.kernel.assign(pytorch_state_dict[f"{h}.mlp.{fc}.weight"].T)
            layer.bias.assign(pytorch_state_dict[f"{h}.mlp.{fc}.bias"])


for model_config in model_configs:
    print(f"\n{'=' * 60}")
    print(f"Converting {model_config['hf_model_name']}...")
    print(f"{'=' * 60}")

    image_size: int = model_config["image_size"]

    keras_model: keras.Model = model_config["keras_model_cls"](weights=None)

    torch_model: torch.nn.Module = OwlViTForObjectDetection.from_pretrained(
        model_config["hf_model_name"],
        token=os.environ.get("HF_TOKEN"),
    ).eval()

    pytorch_state_dict: Dict[str, np.ndarray] = _load_state_dict(
        model_config["hf_model_name"]
    )

    vision_layers: int = keras_model.vision_num_hidden_layers
    text_layers: int = owlvit.OwlViT.TEXT_NUM_HIDDEN_LAYERS

    embed = _walk(keras_model, "vision_model_embeddings")
    embed.class_embedding.assign(
        pytorch_state_dict["owlvit.vision_model.embeddings.class_embedding"]
    )
    embed.patch_embedding.kernel.assign(
        np.transpose(
            pytorch_state_dict["owlvit.vision_model.embeddings.patch_embedding.weight"],
            (2, 3, 1, 0),
        )
    )
    embed.position_embedding.weights[0].assign(
        pytorch_state_dict["owlvit.vision_model.embeddings.position_embedding.weight"]
    )

    for ln_name in ("pre_layernorm", "post_layernorm"):
        layer = _walk(keras_model, f"vision_model_{ln_name}")
        layer.weights[0].assign(
            pytorch_state_dict[f"owlvit.vision_model.{ln_name}.weight"]
        )
        layer.weights[1].assign(
            pytorch_state_dict[f"owlvit.vision_model.{ln_name}.bias"]
        )

    _transfer_encoder(
        keras_model,
        pytorch_state_dict,
        "owlvit.vision_model",
        "vision_model",
        vision_layers,
        "Transferring vision encoder weights",
    )

    text_embed = _walk(keras_model, "text_model_embeddings")
    text_embed.token_embedding.weights[0].assign(
        pytorch_state_dict["owlvit.text_model.embeddings.token_embedding.weight"]
    )
    text_embed.position_embedding.weights[0].assign(
        pytorch_state_dict["owlvit.text_model.embeddings.position_embedding.weight"]
    )
    final_ln = _walk(keras_model, "text_model_final_layer_norm")
    final_ln.weights[0].assign(
        pytorch_state_dict["owlvit.text_model.final_layer_norm.weight"]
    )
    final_ln.weights[1].assign(
        pytorch_state_dict["owlvit.text_model.final_layer_norm.bias"]
    )

    _transfer_encoder(
        keras_model,
        pytorch_state_dict,
        "owlvit.text_model",
        "text_model",
        text_layers,
        "Transferring text encoder weights",
    )

    text_proj = _walk(keras_model, "text_projection")
    text_proj.kernel.assign(pytorch_state_dict["owlvit.text_projection.weight"].T)

    for d in ("dense0", "dense1", "dense2"):
        layer = _walk(keras_model, f"box_head_{d}")
        layer.kernel.assign(pytorch_state_dict[f"box_head.{d}.weight"].T)
        layer.bias.assign(pytorch_state_dict[f"box_head.{d}.bias"])
    for d in ("dense0", "logit_shift", "logit_scale"):
        layer = _walk(keras_model, f"class_head_{d}")
        layer.kernel.assign(pytorch_state_dict[f"class_head.{d}.weight"].T)
        layer.bias.assign(pytorch_state_dict[f"class_head.{d}.bias"])

    top_ln = _walk(keras_model, "layer_norm")
    top_ln.weights[0].assign(pytorch_state_dict["layer_norm.weight"])
    top_ln.weights[1].assign(pytorch_state_dict["layer_norm.bias"])

    print("\nVerifying model equivalence...")

    rng = np.random.default_rng(42)
    img_np = rng.integers(0, 255, size=(image_size, image_size, 3), dtype=np.uint8)
    image = Image.fromarray(img_np)
    text_queries = [["a photo of a cat", "a photo of a dog"]]

    hf_processor = OwlViTProcessor.from_pretrained(
        model_config["hf_model_name"],
        token=os.environ.get("HF_TOKEN"),
    )
    hf_inputs = hf_processor(text=text_queries, images=image, return_tensors="pt")

    with torch.no_grad():
        hf_output = torch_model(
            input_ids=hf_inputs["input_ids"],
            pixel_values=hf_inputs["pixel_values"],
            attention_mask=hf_inputs.get("attention_mask"),
        )
        hf_logits = hf_output.logits.cpu().numpy()
        hf_boxes = hf_output.pred_boxes.cpu().numpy()

    pix_chw = hf_inputs["pixel_values"].cpu().numpy()
    pix_hwc = np.transpose(pix_chw, (0, 2, 3, 1))
    keras_inputs = {
        "pixel_values": keras.ops.convert_to_tensor(pix_hwc, dtype="float32"),
        "input_ids": keras.ops.convert_to_tensor(
            hf_inputs["input_ids"].cpu().numpy(), dtype="int32"
        ),
    }
    keras_output = keras_model(keras_inputs)
    keras_logits = keras.ops.convert_to_numpy(keras_output["logits"])
    keras_boxes = keras.ops.convert_to_numpy(keras_output["pred_boxes"])

    logits_diff = float(np.max(np.abs(hf_logits - keras_logits)))
    boxes_diff = float(np.max(np.abs(hf_boxes - keras_boxes)))

    print(f"Max logits diff:  {logits_diff:.6f}")
    print(f"Max boxes diff:   {boxes_diff:.6f}")

    if logits_diff > 1e-3 or boxes_diff > 1e-3:
        raise ValueError(
            "Model equivalence test failed - model outputs do not match "
            f"(logits: {logits_diff:.6f}, boxes: {boxes_diff:.6f})"
        )

    print("Model equivalence test passed!")

    model_filename: str = (
        f"{model_config['hf_model_name'].split('/')[-1].replace('-', '_')}.weights.h5"
    )
    keras_model.save_weights(model_filename)
    print(f"Model saved successfully as {model_filename}")

    del keras_model, torch_model, pytorch_state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
