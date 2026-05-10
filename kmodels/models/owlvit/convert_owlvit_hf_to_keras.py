import os
from typing import Any, Dict, List

import keras
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from kmodels.models import owlvit
from kmodels.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kmodels.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_nested_layer_weights,
    transfer_weights,
)

weight_name_mapping: Dict[str, str] = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "embeddings": "weight",
}

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

    pytorch_state_dict: Dict[str, np.ndarray] = {
        k: v.cpu().numpy() for k, v in torch_model.state_dict().items()
    }

    vision_layers: int = keras_model.vision_num_hidden_layers
    text_layers: int = owlvit.OwlViT.TEXT_NUM_HIDDEN_LAYERS

    embed = keras_model.get_layer("vision_model_embeddings")
    cls_torch_name = "owlvit.vision_model.embeddings.class_embedding"
    if cls_torch_name not in pytorch_state_dict:
        raise WeightMappingError(
            "vision_model_embeddings/class_embedding", cls_torch_name
        )
    cls_torch = pytorch_state_dict[cls_torch_name]
    if not compare_keras_torch_names(
        "class_embedding", embed.class_embedding, cls_torch_name, cls_torch
    ):
        raise WeightShapeMismatchError(
            "vision_model_embeddings/class_embedding",
            tuple(embed.class_embedding.shape),
            cls_torch_name,
            cls_torch.shape,
        )
    embed.class_embedding.assign(cls_torch)

    patch_torch_name = "owlvit.vision_model.embeddings.patch_embedding.weight"
    if patch_torch_name not in pytorch_state_dict:
        raise WeightMappingError(
            "vision_model_embeddings/patch_embedding/kernel", patch_torch_name
        )
    transfer_weights(
        "conv_kernel",
        embed.patch_embedding.kernel,
        pytorch_state_dict[patch_torch_name],
    )

    pos_torch_name = "owlvit.vision_model.embeddings.position_embedding.weight"
    if pos_torch_name not in pytorch_state_dict:
        raise WeightMappingError(
            "vision_model_embeddings/position_embedding/embeddings", pos_torch_name
        )
    transfer_weights(
        "position_embedding/embeddings",
        embed.position_embedding.weights[0],
        pytorch_state_dict[pos_torch_name],
    )

    transfer_nested_layer_weights(
        keras_layer=keras_model.get_layer("vision_model_pre_layernorm"),
        torch_weights_dict=pytorch_state_dict,
        torch_prefix="owlvit.vision_model.pre_layernorm",
        name_mapping=weight_name_mapping,
    )
    transfer_nested_layer_weights(
        keras_layer=keras_model.get_layer("vision_model_post_layernorm"),
        torch_weights_dict=pytorch_state_dict,
        torch_prefix="owlvit.vision_model.post_layernorm",
        name_mapping=weight_name_mapping,
    )

    for i in tqdm(range(vision_layers), desc="Transferring vision encoder weights"):
        kp = f"vision_model_layers_{i}"
        tp = f"owlvit.vision_model.encoder.layers.{i}"
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"{kp}_self_attn"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"{tp}.self_attn",
            name_mapping=weight_name_mapping,
        )
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"{kp}_layer_norm1"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"{tp}.layer_norm1",
            name_mapping=weight_name_mapping,
        )
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"{kp}_layer_norm2"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"{tp}.layer_norm2",
            name_mapping=weight_name_mapping,
        )
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"{kp}_mlp_fc1"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"{tp}.mlp.fc1",
            name_mapping=weight_name_mapping,
        )
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"{kp}_mlp_fc2"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"{tp}.mlp.fc2",
            name_mapping=weight_name_mapping,
        )

    transfer_nested_layer_weights(
        keras_layer=keras_model.get_layer("text_model_embeddings"),
        torch_weights_dict=pytorch_state_dict,
        torch_prefix="owlvit.text_model.embeddings",
        name_mapping=weight_name_mapping,
    )
    transfer_nested_layer_weights(
        keras_layer=keras_model.get_layer("text_model_final_layer_norm"),
        torch_weights_dict=pytorch_state_dict,
        torch_prefix="owlvit.text_model.final_layer_norm",
        name_mapping=weight_name_mapping,
    )

    for i in tqdm(range(text_layers), desc="Transferring text encoder weights"):
        kp = f"text_model_layers_{i}"
        tp = f"owlvit.text_model.encoder.layers.{i}"
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"{kp}_self_attn"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"{tp}.self_attn",
            name_mapping=weight_name_mapping,
        )
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"{kp}_layer_norm1"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"{tp}.layer_norm1",
            name_mapping=weight_name_mapping,
        )
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"{kp}_layer_norm2"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"{tp}.layer_norm2",
            name_mapping=weight_name_mapping,
        )
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"{kp}_mlp_fc1"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"{tp}.mlp.fc1",
            name_mapping=weight_name_mapping,
        )
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"{kp}_mlp_fc2"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"{tp}.mlp.fc2",
            name_mapping=weight_name_mapping,
        )

    transfer_nested_layer_weights(
        keras_layer=keras_model.get_layer("text_projection"),
        torch_weights_dict=pytorch_state_dict,
        torch_prefix="owlvit.text_projection",
        name_mapping=weight_name_mapping,
    )

    for d in ("dense0", "dense1", "dense2"):
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"box_head_{d}"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"box_head.{d}",
            name_mapping=weight_name_mapping,
        )
    for d in ("dense0", "logit_shift", "logit_scale"):
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"class_head_{d}"),
            torch_weights_dict=pytorch_state_dict,
            torch_prefix=f"class_head.{d}",
            name_mapping=weight_name_mapping,
        )
    transfer_nested_layer_weights(
        keras_layer=keras_model.get_layer("layer_norm"),
        torch_weights_dict=pytorch_state_dict,
        torch_prefix="layer_norm",
        name_mapping=weight_name_mapping,
    )

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
    input_ids_np = hf_inputs["input_ids"].cpu().numpy()
    del torch_model, hf_output, hf_inputs
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pix_hwc = np.transpose(pix_chw, (0, 2, 3, 1))
    keras_inputs = {
        "pixel_values": keras.ops.convert_to_tensor(pix_hwc, dtype="float32"),
        "input_ids": keras.ops.convert_to_tensor(input_ids_np, dtype="int32"),
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

    del keras_model, pytorch_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
