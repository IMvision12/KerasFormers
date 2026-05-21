import gc
import os
from typing import Any, Dict, List

import keras
import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from kerasformers.models.mask2former import Mask2FormerUniversalSegment
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights


def transfer_mask2former_weights(keras_model, hf_state_dict):
    """Transfer all Mask2Former weights into a Keras Mask2FormerUniversalSegment."""

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

    backbone_prefix = "model.pixel_level_module.encoder"
    pixel_decoder_prefix = "model.pixel_level_module.decoder"
    transformer_prefix = "model.transformer_module"

    # ---- Swin backbone ----
    print("Transferring Swin backbone...", flush=True)
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

    for i in range(len(backbone.hidden_states_norms)):
        assign_ln(
            backbone.hidden_states_norms[i],
            sd[f"{backbone_prefix}.hidden_states_norms.stage{i + 1}.weight"],
            sd[f"{backbone_prefix}.hidden_states_norms.stage{i + 1}.bias"],
        )

    # ---- Pixel decoder (MSDeformAttn) ----
    print("Transferring pixel decoder...", flush=True)
    for i in range(3):
        p_proj = f"{pixel_decoder_prefix}.input_projections.{i}"
        assign_conv(
            keras_model.get_layer(f"pixel_decoder_input_projections_{i}_conv"),
            sd[f"{p_proj}.0.weight"],
            sd[f"{p_proj}.0.bias"],
        )
        assign_ln(
            keras_model.get_layer(f"pixel_decoder_input_projections_{i}_norm"),
            sd[f"{p_proj}.1.weight"],
            sd[f"{p_proj}.1.bias"],
        )

    for i in range(keras_model.encoder_layers):
        p = f"{pixel_decoder_prefix}.encoder.layers.{i}"
        prefix_k = f"pixel_decoder_encoder_layers_{i}"
        attn = keras_model.get_layer(f"{prefix_k}_self_attn")
        assign_dense(
            attn.sampling_offsets,
            sd[f"{p}.self_attn.sampling_offsets.weight"],
            sd[f"{p}.self_attn.sampling_offsets.bias"],
        )
        assign_dense(
            attn.attention_weights,
            sd[f"{p}.self_attn.attention_weights.weight"],
            sd[f"{p}.self_attn.attention_weights.bias"],
        )
        assign_dense(
            attn.value_proj,
            sd[f"{p}.self_attn.value_proj.weight"],
            sd[f"{p}.self_attn.value_proj.bias"],
        )
        assign_dense(
            attn.output_proj,
            sd[f"{p}.self_attn.output_proj.weight"],
            sd[f"{p}.self_attn.output_proj.bias"],
        )
        assign_ln(
            keras_model.get_layer(f"{prefix_k}_self_attn_layer_norm"),
            sd[f"{p}.self_attn_layer_norm.weight"],
            sd[f"{p}.self_attn_layer_norm.bias"],
        )
        assign_dense(
            keras_model.get_layer(f"{prefix_k}_fc1"),
            sd[f"{p}.fc1.weight"],
            sd[f"{p}.fc1.bias"],
        )
        assign_dense(
            keras_model.get_layer(f"{prefix_k}_fc2"),
            sd[f"{p}.fc2.weight"],
            sd[f"{p}.fc2.bias"],
        )
        assign_ln(
            keras_model.get_layer(f"{prefix_k}_final_layer_norm"),
            sd[f"{p}.final_layer_norm.weight"],
            sd[f"{p}.final_layer_norm.bias"],
        )

    # Pixel decoder level_embed (separate from transformer module's level_embed).
    keras_model.get_layer("pixel_decoder_level_embed").weight.assign(
        sd[f"{pixel_decoder_prefix}.level_embed"]
    )

    assign_conv(
        keras_model.get_layer("pixel_decoder_adapter_1_conv"),
        sd[f"{pixel_decoder_prefix}.adapter_1.0.weight"],
    )
    assign_ln(
        keras_model.get_layer("pixel_decoder_adapter_1_norm"),
        sd[f"{pixel_decoder_prefix}.adapter_1.1.weight"],
        sd[f"{pixel_decoder_prefix}.adapter_1.1.bias"],
    )
    assign_conv(
        keras_model.get_layer("pixel_decoder_layer_1_conv"),
        sd[f"{pixel_decoder_prefix}.layer_1.0.weight"],
    )
    assign_ln(
        keras_model.get_layer("pixel_decoder_layer_1_norm"),
        sd[f"{pixel_decoder_prefix}.layer_1.1.weight"],
        sd[f"{pixel_decoder_prefix}.layer_1.1.bias"],
    )
    assign_conv(
        keras_model.get_layer("pixel_decoder_mask_projection"),
        sd[f"{pixel_decoder_prefix}.mask_projection.weight"],
        sd[f"{pixel_decoder_prefix}.mask_projection.bias"],
    )

    # ---- Transformer decoder (masked attention) ----
    print("Transferring transformer decoder...", flush=True)
    keras_model.get_layer("transformer_decoder_queries_features").weight.assign(
        sd[f"{transformer_prefix}.queries_features.weight"]
    )
    keras_model.get_layer("transformer_decoder_queries_embedder").weight.assign(
        sd[f"{transformer_prefix}.queries_embedder.weight"]
    )
    keras_model.get_layer("transformer_decoder_level_embed").weight.assign(
        sd[f"{transformer_prefix}.level_embed.weight"]
    )

    for i in range(keras_model.decoder_layers):
        p = f"{transformer_prefix}.decoder.layers.{i}"
        prefix_k = f"transformer_decoder_layers_{i}"

        # Self-attention. The output projection is ``out_proj`` in current
        # transformers and ``o_proj`` in the refactored attention; pick whichever
        # the checkpoint carries.
        sa = keras_model.get_layer(f"{prefix_k}_self_attn")
        hf_out_proj = "o_proj" if f"{p}.self_attn.o_proj.weight" in sd else "out_proj"
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            hf_proj = hf_out_proj if proj == "out_proj" else proj
            assign_dense(
                getattr(sa, proj),
                sd[f"{p}.self_attn.{hf_proj}.weight"],
                sd[f"{p}.self_attn.{hf_proj}.bias"],
            )
        assign_ln(
            keras_model.get_layer(f"{prefix_k}_self_attn_layer_norm"),
            sd[f"{p}.self_attn_layer_norm.weight"],
            sd[f"{p}.self_attn_layer_norm.bias"],
        )

        # Cross-attention (HF uses a fused in_proj_weight = (3*hidden, hidden)).
        ca = keras_model.get_layer(f"{prefix_k}_cross_attn")
        ca.in_proj_weight.assign(sd[f"{p}.cross_attn.in_proj_weight"])
        ca.in_proj_bias.assign(sd[f"{p}.cross_attn.in_proj_bias"])
        assign_dense(
            ca.out_proj,
            sd[f"{p}.cross_attn.out_proj.weight"],
            sd[f"{p}.cross_attn.out_proj.bias"],
        )
        assign_ln(
            keras_model.get_layer(f"{prefix_k}_cross_attn_layer_norm"),
            sd[f"{p}.cross_attn_layer_norm.weight"],
            sd[f"{p}.cross_attn_layer_norm.bias"],
        )

        # FFN
        assign_dense(
            keras_model.get_layer(f"{prefix_k}_fc1"),
            sd[f"{p}.fc1.weight"],
            sd[f"{p}.fc1.bias"],
        )
        assign_dense(
            keras_model.get_layer(f"{prefix_k}_fc2"),
            sd[f"{p}.fc2.weight"],
            sd[f"{p}.fc2.bias"],
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

    # Mask embedder (3-layer MLP)
    for i in range(3):
        assign_dense(
            keras_model.get_layer(f"transformer_decoder_mask_embedder_{i}"),
            sd[
                f"{transformer_prefix}.decoder.mask_predictor.mask_embedder.{i}.0.weight"
            ],
            sd[f"{transformer_prefix}.decoder.mask_predictor.mask_embedder.{i}.0.bias"],
        )

    # ---- class predictor ----
    print("Transferring class_predictor...", flush=True)
    assign_dense(
        keras_model.get_layer("class_predictor"),
        sd["class_predictor.weight"],
        sd["class_predictor.bias"],
    )


MASK2FORMER_CONVERSION_CONFIG: List[Dict[str, Any]] = [
    {
        "variant": "mask2former-swin-tiny-coco-instance",
        "hf_id": "facebook/mask2former-swin-tiny-coco-instance",
    },
]


if __name__ == "__main__":
    # Run the HF reference on the same device Keras (torch backend) uses, so the
    # diff reflects the conversion rather than GPU-vs-CPU float accumulation.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for cfg in MASK2FORMER_CONVERSION_CONFIG:
        variant = cfg["variant"]
        hf_id = cfg["hf_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        keras_model = Mask2FormerUniversalSegment.from_weights(
            variant, load_weights=False
        )

        hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            hf_id, token=os.environ.get("HF_TOKEN")
        ).eval()
        sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
        hf_model = hf_model.to(device)

        transfer_mask2former_weights(keras_model, sd)

        input_size = keras_model.input_image_shape[0]
        rng = np.random.default_rng(42)
        img_np = rng.integers(0, 255, size=(input_size, input_size, 3), dtype=np.uint8)
        from PIL import Image as PILImage

        image = PILImage.fromarray(img_np)
        hf_proc = Mask2FormerImageProcessor.from_pretrained(hf_id)
        hf_inputs = hf_proc(images=image, return_tensors="pt")

        with torch.no_grad():
            hf_out = hf_model(pixel_values=hf_inputs["pixel_values"].to(device))
            hf_class = hf_out.class_queries_logits.cpu().numpy()
            hf_mask = hf_out.masks_queries_logits.cpu().numpy()

        pix_chw = hf_inputs["pixel_values"].cpu().numpy()
        del hf_model, hf_out, hf_inputs, sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pix_hwc = np.transpose(pix_chw, (0, 2, 3, 1))
        if pix_hwc.shape[1] != input_size or pix_hwc.shape[2] != input_size:
            new_img = np.zeros((1, input_size, input_size, 3), dtype=np.float32)
            h, w = pix_hwc.shape[1], pix_hwc.shape[2]
            new_img[0, :h, :w, :] = pix_hwc[0]
            pix_hwc = new_img

        keras_input = keras.ops.convert_to_tensor(pix_hwc, dtype="float32")
        keras_out = keras_model(keras_input)
        keras_class = keras.ops.convert_to_numpy(keras_out["class_queries_logits"])
        keras_mask = keras.ops.convert_to_numpy(keras_out["masks_queries_logits"])

        class_diff = float(np.max(np.abs(hf_class - keras_class)))
        mask_diff = float(np.max(np.abs(hf_mask - keras_mask)))
        print(f"  Max class logits diff: {class_diff:.6f}", flush=True)
        print(f"  Max mask logits diff:  {mask_diff:.6f}", flush=True)

        out_filename = f"{variant}.weights.h5"
        keras_model.save_weights(out_filename)
        print(f"  Saved -> {out_filename}", flush=True)

        del keras_model
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
