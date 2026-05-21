import gc
import os
from typing import Any, Dict, List

import keras
import numpy as np
import torch
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor

from kerasformers.models.maskformer import MaskFormerUniversalSegment
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights


def transfer_maskformer_weights(keras_model, hf_state_dict):
    """Transfer all HF MaskFormer weights into a Keras MaskFormerUniversalSegment model."""

    def assign_dense(keras_layer, weight_arr, bias_arr=None):
        transfer_weights("kernel", keras_layer.weights[0], weight_arr)
        if bias_arr is not None:
            keras_layer.weights[1].assign(bias_arr)

    def assign_conv(keras_layer, weight_arr, bias_arr=None):
        transfer_weights("conv_kernel", keras_layer.weights[0], weight_arr)
        if bias_arr is not None:
            keras_layer.weights[1].assign(bias_arr)

    def assign_ln(keras_layer, weight_arr, bias_arr):
        keras_layer.weights[0].assign(weight_arr)
        keras_layer.weights[1].assign(bias_arr)

    sd = hf_state_dict
    backbone = keras_model.get_layer("backbone")

    sample_key = next(iter(sd))
    model_prefix = "model." if sample_key.startswith("model.") else ""

    backbone_prefix = f"{model_prefix}pixel_level_module.encoder.model"
    pixel_decoder_prefix = f"{model_prefix}pixel_level_module.decoder"
    transformer_prefix = f"{model_prefix}transformer_module"

    print("Transferring Swin backbone...")
    assign_conv(
        backbone.patch_embeddings.projection,
        sd[f"{backbone_prefix}.embeddings.patch_embeddings.projection.weight"],
        sd[f"{backbone_prefix}.embeddings.patch_embeddings.projection.bias"],
    )
    assign_ln(
        backbone.embeddings_norm,
        sd[f"{backbone_prefix}.embeddings.norm.weight"],
        sd[f"{backbone_prefix}.embeddings.norm.bias"],
    )

    for stage_idx, stage in enumerate(backbone.stages):
        for block_idx, block in enumerate(stage.blocks):
            p = f"{backbone_prefix}.encoder.layers.{stage_idx}.blocks.{block_idx}"
            assign_ln(
                block.layernorm_before,
                sd[f"{p}.layernorm_before.weight"],
                sd[f"{p}.layernorm_before.bias"],
            )
            assign_dense(
                block.attention.self_attn.query,
                sd[f"{p}.attention.self.query.weight"],
                sd[f"{p}.attention.self.query.bias"],
            )
            assign_dense(
                block.attention.self_attn.key,
                sd[f"{p}.attention.self.key.weight"],
                sd[f"{p}.attention.self.key.bias"],
            )
            assign_dense(
                block.attention.self_attn.value,
                sd[f"{p}.attention.self.value.weight"],
                sd[f"{p}.attention.self.value.bias"],
            )
            block.attention.self_attn.relative_position_bias_table.assign(
                sd[f"{p}.attention.self.relative_position_bias_table"]
            )
            assign_dense(
                block.attention.output_dense,
                sd[f"{p}.attention.output.dense.weight"],
                sd[f"{p}.attention.output.dense.bias"],
            )
            assign_ln(
                block.layernorm_after,
                sd[f"{p}.layernorm_after.weight"],
                sd[f"{p}.layernorm_after.bias"],
            )
            assign_dense(
                block.intermediate_dense,
                sd[f"{p}.intermediate.dense.weight"],
                sd[f"{p}.intermediate.dense.bias"],
            )
            assign_dense(
                block.output_dense,
                sd[f"{p}.output.dense.weight"],
                sd[f"{p}.output.dense.bias"],
            )

        if stage.downsample is not None:
            ds_prefix = f"{backbone_prefix}.encoder.layers.{stage_idx}.downsample"
            transfer_weights(
                "kernel",
                stage.downsample.reduction.weights[0],
                sd[f"{ds_prefix}.reduction.weight"],
            )
            assign_ln(
                stage.downsample.norm,
                sd[f"{ds_prefix}.norm.weight"],
                sd[f"{ds_prefix}.norm.bias"],
            )

    encoder_prefix = backbone_prefix.rsplit(".model", 1)[0]
    for i in range(len(backbone.hidden_states_norms)):
        assign_ln(
            backbone.hidden_states_norms[i],
            sd[f"{encoder_prefix}.hidden_states_norms.{i}.weight"],
            sd[f"{encoder_prefix}.hidden_states_norms.{i}.bias"],
        )

    print("Transferring pixel decoder...")
    assign_conv(
        keras_model.get_layer("pixel_decoder_fpn_stem_conv"),
        sd[f"{pixel_decoder_prefix}.fpn.stem.0.weight"],
    )
    assign_ln(
        keras_model.get_layer("pixel_decoder_fpn_stem_norm"),
        sd[f"{pixel_decoder_prefix}.fpn.stem.1.weight"],
        sd[f"{pixel_decoder_prefix}.fpn.stem.1.bias"],
    )

    for i in range(3):
        p = f"{pixel_decoder_prefix}.fpn.layers.{i}"
        assign_conv(
            keras_model.get_layer(f"pixel_decoder_fpn_layer_{i}_proj_conv"),
            sd[f"{p}.proj.0.weight"],
        )
        assign_ln(
            keras_model.get_layer(f"pixel_decoder_fpn_layer_{i}_proj_norm"),
            sd[f"{p}.proj.1.weight"],
            sd[f"{p}.proj.1.bias"],
        )
        assign_conv(
            keras_model.get_layer(f"pixel_decoder_fpn_layer_{i}_block_conv"),
            sd[f"{p}.block.0.weight"],
        )
        assign_ln(
            keras_model.get_layer(f"pixel_decoder_fpn_layer_{i}_block_norm"),
            sd[f"{p}.block.1.weight"],
            sd[f"{p}.block.1.bias"],
        )

    assign_conv(
        keras_model.get_layer("pixel_decoder_mask_projection"),
        sd[f"{pixel_decoder_prefix}.mask_projection.weight"],
        sd[f"{pixel_decoder_prefix}.mask_projection.bias"],
    )

    print("Transferring transformer decoder...")
    keras_model.get_layer("transformer_decoder_queries_embedder").weight.assign(
        sd[f"{transformer_prefix}.queries_embedder.weight"]
    )
    assign_conv(
        keras_model.get_layer("transformer_decoder_input_projection"),
        sd[f"{transformer_prefix}.input_projection.weight"],
        sd[f"{transformer_prefix}.input_projection.bias"],
    )

    for i in range(keras_model.decoder_layers):
        p = f"{transformer_prefix}.decoder.layers.{i}"
        prefix_k = f"transformer_decoder_layers_{i}"

        for attn_name in ("self_attn", "encoder_attn"):
            attn = keras_model.get_layer(f"{prefix_k}_{attn_name}")
            hf_out_proj = (
                "o_proj" if f"{p}.{attn_name}.o_proj.weight" in sd else "out_proj"
            )
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                hf_proj = hf_out_proj if proj == "o_proj" else proj
                assign_dense(
                    getattr(attn, proj),
                    sd[f"{p}.{attn_name}.{hf_proj}.weight"],
                    sd[f"{p}.{attn_name}.{hf_proj}.bias"],
                )
            assign_ln(
                keras_model.get_layer(f"{prefix_k}_{attn_name}_layer_norm"),
                sd[f"{p}.{attn_name}_layer_norm.weight"],
                sd[f"{p}.{attn_name}_layer_norm.bias"],
            )

        assign_dense(
            keras_model.get_layer(f"{prefix_k}_fc1"),
            sd[f"{p}.mlp.fc1.weight"],
            sd[f"{p}.mlp.fc1.bias"],
        )
        assign_dense(
            keras_model.get_layer(f"{prefix_k}_fc2"),
            sd[f"{p}.mlp.fc2.weight"],
            sd[f"{p}.mlp.fc2.bias"],
        )
        assign_ln(
            keras_model.get_layer(f"{prefix_k}_final_layer_norm"),
            sd[f"{p}.final_layer_norm.weight"],
            sd[f"{p}.final_layer_norm.bias"],
        )

    assign_ln(
        keras_model.get_layer("transformer_decoder_layernorm"),
        sd[f"{transformer_prefix}.decoder.layernorm.weight"],
        sd[f"{transformer_prefix}.decoder.layernorm.bias"],
    )

    print("Transferring heads...")
    assign_dense(
        keras_model.get_layer("class_predictor"),
        sd["class_predictor.weight"],
        sd["class_predictor.bias"],
    )
    for i in range(3):
        assign_dense(
            keras_model.get_layer(f"mask_embedder_{i}"),
            sd[f"mask_embedder.{i}.0.weight"],
            sd[f"mask_embedder.{i}.0.bias"],
        )


MASKFORMER_CONVERSION_CONFIG: List[Dict[str, Any]] = [
    {
        "variant": "maskformer-swin-tiny-ade",
        "hf_id": "facebook/maskformer-swin-tiny-ade",
    },
    {
        "variant": "maskformer-swin-tiny-coco",
        "hf_id": "facebook/maskformer-swin-tiny-coco",
    },
    {
        "variant": "maskformer-swin-small-coco",
        "hf_id": "facebook/maskformer-swin-small-coco",
    },
    {
        "variant": "maskformer-swin-base-ade",
        "hf_id": "facebook/maskformer-swin-base-ade",
    },
    {
        "variant": "maskformer-swin-base-coco",
        "hf_id": "facebook/maskformer-swin-base-coco",
    },
]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for cfg in MASKFORMER_CONVERSION_CONFIG:
        variant = cfg["variant"]
        hf_id = cfg["hf_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        keras_model = MaskFormerUniversalSegment.from_weights(
            variant, load_weights=False
        )

        hf_model = MaskFormerForInstanceSegmentation.from_pretrained(
            hf_id, token=os.environ.get("HF_TOKEN")
        ).eval()
        sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
        hf_model = hf_model.to(device)

        transfer_maskformer_weights(keras_model, sd)

        input_image_shape = keras_model.input_image_shape
        rng = np.random.default_rng(42)
        img_np = rng.integers(
            0, 255, size=(input_image_shape[0], input_image_shape[1], 3), dtype=np.uint8
        )
        from PIL import Image as PILImage

        image = PILImage.fromarray(img_np)
        hf_proc = MaskFormerImageProcessor.from_pretrained(hf_id)
        hf_inputs = hf_proc(images=image, return_tensors="pt")

        with torch.no_grad():
            hf_out = hf_model(pixel_values=hf_inputs["pixel_values"].to(device))
            hf_class_logits = hf_out.class_queries_logits.cpu().numpy()
            hf_mask_logits = hf_out.masks_queries_logits.cpu().numpy()

        pix_chw = hf_inputs["pixel_values"].cpu().numpy()
        del hf_model, hf_out, hf_inputs, sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pix_hwc = np.transpose(pix_chw, (0, 2, 3, 1))
        if pix_hwc.shape[1] != input_image_shape[0]:
            new_img = np.zeros(
                (1, input_image_shape[0], input_image_shape[1], 3), dtype=np.float32
            )
            h, w = pix_hwc.shape[1], pix_hwc.shape[2]
            new_img[0, :h, :w, :] = pix_hwc[0]
            pix_hwc = new_img

        keras_input = keras.ops.convert_to_tensor(pix_hwc, dtype="float32")
        keras_out = keras_model(keras_input)
        keras_class = keras.ops.convert_to_numpy(keras_out["class_queries_logits"])
        keras_mask = keras.ops.convert_to_numpy(keras_out["masks_queries_logits"])

        class_diff = float(np.max(np.abs(hf_class_logits - keras_class)))
        mask_diff = float(np.max(np.abs(hf_mask_logits - keras_mask)))
        print(f"  Max class logits diff: {class_diff:.6f}")
        print(f"  Max mask logits diff:  {mask_diff:.6f}")

        if class_diff > 5e-3 or mask_diff > 1e-2:
            raise ValueError(
                f"{variant}: class {class_diff:.6f}, mask {mask_diff:.6f} exceed tolerance"
            )

        out_filename = f"{variant}.weights.h5"
        keras_model.save_weights(out_filename)
        print(f"  Saved -> {out_filename}")

        del keras_model
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
