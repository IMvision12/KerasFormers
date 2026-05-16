from typing import Dict, List

import numpy as np
from tqdm import tqdm

from kerasformers.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kerasformers.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_nested_layer_weights,
    transfer_weights,
)

backbone_weight_name_mapping: Dict[str, str] = {
    "backbone_layer": "model.backbone.model.layer",
    "_": ".",
    "downsample.conv": "downsample.0",
    "downsample.bn": "downsample.1",
    "backbone.conv1": "model.backbone.model.conv1",
    "backbone.bn1": "model.backbone.model.bn1",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
}


def transfer_detr_weights(keras_model, state_dict):
    """Transfer DETR weights from a ``DetrForObjectDetection`` state_dict.

    Handles the ResNet backbone, the input projection, the transformer
    encoder/decoder, the decoder layer norm, the classifier head, and
    the bbox MLP head.

    Args:
        keras_model: A ``DETRDetect`` instance.
        state_dict: Mapping of torch weight names to numpy arrays from
            ``DetrForObjectDetection.state_dict()``.
    """
    backbone_layers = [
        layer for layer in keras_model.layers if layer.name.startswith("backbone_")
    ]

    backbone_trainable = []
    backbone_non_trainable = []
    for layer in backbone_layers:
        for weight in layer.trainable_weights:
            backbone_trainable.append((weight, f"{layer.name}_{weight.name}"))
        for weight in layer.non_trainable_weights:
            backbone_non_trainable.append((weight, f"{layer.name}_{weight.name}"))

    for keras_weight, keras_weight_name in tqdm(
        backbone_trainable + backbone_non_trainable,
        total=len(backbone_trainable + backbone_non_trainable),
        desc="Transferring backbone weights",
    ):
        torch_weight_name: str = keras_weight_name
        for keras_name_part, torch_name_part in backbone_weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(
                keras_name_part, torch_name_part
            )

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

    input_proj = keras_model.get_layer("input_projection")
    conv_w = state_dict["model.input_projection.weight"]
    input_proj.weights[0].assign(np.transpose(conv_w, (2, 3, 1, 0)))
    input_proj.weights[1].assign(state_dict["model.input_projection.bias"])

    query_layer = keras_model.get_layer("query_position_embeddings")
    query_layer.weights[0].assign(state_dict["model.query_position_embeddings.weight"])

    ln_mapping = {"gamma": "weight", "beta": "bias"}
    dense_mapping = {"kernel": "weight"}

    for i in tqdm(
        range(keras_model.num_encoder_layers), desc="Transferring encoder weights"
    ):
        hf_prefix = f"model.encoder.layers.{i}"
        k_prefix = f"encoder_layers_{i}"

        sa_mapping = {
            f"{k_prefix}_self_attn_": "",
            "out_proj": "o_proj",
            "kernel": "weight",
        }
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_self_attn"),
            state_dict,
            f"{hf_prefix}.self_attn",
            name_mapping=sa_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_self_attn_layer_norm"),
            state_dict,
            f"{hf_prefix}.self_attn_layer_norm",
            name_mapping=ln_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_fc1"),
            state_dict,
            f"{hf_prefix}.mlp.fc1",
            name_mapping=dense_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_fc2"),
            state_dict,
            f"{hf_prefix}.mlp.fc2",
            name_mapping=dense_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_final_layer_norm"),
            state_dict,
            f"{hf_prefix}.final_layer_norm",
            name_mapping=ln_mapping,
        )

    for i in tqdm(
        range(keras_model.num_decoder_layers), desc="Transferring decoder weights"
    ):
        hf_prefix = f"model.decoder.layers.{i}"
        k_prefix = f"decoder_layers_{i}"

        sa_mapping = {
            f"{k_prefix}_self_attn_": "",
            "out_proj": "o_proj",
            "kernel": "weight",
        }
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_self_attn"),
            state_dict,
            f"{hf_prefix}.self_attn",
            name_mapping=sa_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_self_attn_layer_norm"),
            state_dict,
            f"{hf_prefix}.self_attn_layer_norm",
            name_mapping=ln_mapping,
        )
        ca_mapping = {
            f"{k_prefix}_encoder_attn_": "",
            "out_proj": "o_proj",
            "kernel": "weight",
        }
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_encoder_attn"),
            state_dict,
            f"{hf_prefix}.encoder_attn",
            name_mapping=ca_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_encoder_attn_layer_norm"),
            state_dict,
            f"{hf_prefix}.encoder_attn_layer_norm",
            name_mapping=ln_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_fc1"),
            state_dict,
            f"{hf_prefix}.mlp.fc1",
            name_mapping=dense_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_fc2"),
            state_dict,
            f"{hf_prefix}.mlp.fc2",
            name_mapping=dense_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_final_layer_norm"),
            state_dict,
            f"{hf_prefix}.final_layer_norm",
            name_mapping=ln_mapping,
        )

    transfer_nested_layer_weights(
        keras_model.get_layer("decoder_layernorm"),
        state_dict,
        "model.decoder.layernorm",
        name_mapping=ln_mapping,
    )

    transfer_nested_layer_weights(
        keras_model.get_layer("class_labels_classifier"),
        state_dict,
        "class_labels_classifier",
        name_mapping=dense_mapping,
    )

    for layer_idx in range(3):
        transfer_nested_layer_weights(
            keras_model.get_layer(f"bbox_predictor_{layer_idx}"),
            state_dict,
            f"bbox_predictor.layers.{layer_idx}",
            name_mapping=dense_mapping,
        )


if __name__ == "__main__":
    import keras
    import torch
    import torchvision.transforms as T
    from transformers import DetrForObjectDetection

    from kerasformers.models.detr import DETRDetect

    model_configs: List[Dict[str, object]] = [
        {
            "variant": "detr-resnet-50",
            "hf_model_name": "facebook/detr-resnet-50",
            "output": "detr_resnet50.weights.h5",
            "input_shape": [800, 800, 3],
            "num_classes": 92,
            "num_queries": 100,
        },
        {
            "variant": "detr-resnet-101",
            "hf_model_name": "facebook/detr-resnet-101",
            "output": "detr_resnet101.weights.h5",
            "input_shape": [800, 800, 3],
            "num_classes": 92,
            "num_queries": 100,
        },
    ]

    for cfg in model_configs:
        print(f"\n{'=' * 60}")
        print(f"Converting {cfg['hf_model_name']}...")
        print(f"{'=' * 60}")

        keras_model = DETRDetect.from_weights(
            cfg["variant"],
            load_weights=False,
            input_shape=cfg["input_shape"],
            num_classes=cfg["num_classes"],
            num_queries=cfg["num_queries"],
            include_normalization=False,
        )

        torch_model: torch.nn.Module = DetrForObjectDetection.from_pretrained(
            cfg["hf_model_name"]
        ).eval()
        pytorch_state_dict: Dict[str, np.ndarray] = {
            k: v.cpu().numpy() for k, v in torch_model.state_dict().items()
        }

        transfer_detr_weights(keras_model, pytorch_state_dict)

        print("\nVerifying model equivalence...")

        np.random.seed(42)
        test_input = np.random.rand(1, 800, 800, 3).astype(np.float32)

        hf_input = torch.tensor(test_input).permute(0, 3, 1, 2)
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        hf_input_norm = normalize(hf_input)

        with torch.no_grad():
            hf_output = torch_model(hf_input_norm)
            hf_logits = hf_output.logits.numpy()
            hf_boxes = hf_output.pred_boxes.numpy()

        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        keras_input_norm = (test_input - mean) / std

        keras_output = keras_model(keras_input_norm.astype(np.float32), training=False)
        keras_logits = keras.ops.convert_to_numpy(keras_output["logits"])
        keras_boxes = keras.ops.convert_to_numpy(keras_output["pred_boxes"])

        logits_diff = np.max(np.abs(hf_logits - keras_logits))
        boxes_diff = np.max(np.abs(hf_boxes - keras_boxes))

        print(f"Max logits diff:  {logits_diff:.6f}")
        print(f"Max boxes diff:   {boxes_diff:.6f}")

        if logits_diff > 1e-3 or boxes_diff > 1e-3:
            raise ValueError(
                "Model equivalence test failed - model outputs do not match "
                f"(logits: {logits_diff:.6f}, boxes: {boxes_diff:.6f})"
            )

        print("Model equivalence test passed!")

        model_filename = cfg["output"]
        keras_model.save_weights(model_filename)
        print(f"Model saved successfully as {model_filename}")

        del keras_model, torch_model, pytorch_state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
