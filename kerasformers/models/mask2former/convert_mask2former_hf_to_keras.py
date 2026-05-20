import gc
import os
from typing import Any, Dict, List

import keras
import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from kerasformers.models.mask2former import Mask2FormerUniversalSegment
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights


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


def transfer_swin_backbone(backbone, sd, prefix):
    """Transfer HF Mask2Former Swin backbone weights into our Keras backbone."""
    assign_conv(
        backbone.patch_embeddings.projection,
        sd[f"{prefix}.embeddings.patch_embeddings.projection.weight"],
        sd[f"{prefix}.embeddings.patch_embeddings.projection.bias"],
    )
    assign_ln(
        backbone.embeddings_norm,
        sd[f"{prefix}.embeddings.norm.weight"],
        sd[f"{prefix}.embeddings.norm.bias"],
    )

    for stage_idx, stage in enumerate(backbone.stages):
        for block_idx, block in enumerate(stage.blocks):
            p = f"{prefix}.encoder.layers.{stage_idx}.blocks.{block_idx}"
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
            ds_prefix = f"{prefix}.encoder.layers.{stage_idx}.downsample"
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
            sd[f"{prefix}.hidden_states_norms.stage{i + 1}.weight"],
            sd[f"{prefix}.hidden_states_norms.stage{i + 1}.bias"],
        )


def transfer_pixel_decoder(keras_model, sd, prefix, encoder_layers):
    """Transfer Mask2Former MSDeformAttn pixel decoder weights."""
    # input_projections.{0,1,2}.0 (Conv) and .1 (GN)
    for i in range(3):
        p_proj = f"{prefix}.input_projections.{i}"
        conv = keras_model.get_layer(f"pixel_decoder_input_projections_{i}_conv")
        norm = keras_model.get_layer(f"pixel_decoder_input_projections_{i}_norm")
        assign_conv(conv, sd[f"{p_proj}.0.weight"], sd[f"{p_proj}.0.bias"])
        assign_ln(norm, sd[f"{p_proj}.1.weight"], sd[f"{p_proj}.1.bias"])

    # MSDeformAttn encoder layers
    for i in range(encoder_layers):
        p = f"{prefix}.encoder.layers.{i}"
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
        sa_ln = keras_model.get_layer(f"{prefix_k}_self_attn_layer_norm")
        assign_ln(
            sa_ln,
            sd[f"{p}.self_attn_layer_norm.weight"],
            sd[f"{p}.self_attn_layer_norm.bias"],
        )
        fc1 = keras_model.get_layer(f"{prefix_k}_fc1")
        fc2 = keras_model.get_layer(f"{prefix_k}_fc2")
        assign_dense(fc1, sd[f"{p}.fc1.weight"], sd[f"{p}.fc1.bias"])
        assign_dense(fc2, sd[f"{p}.fc2.weight"], sd[f"{p}.fc2.bias"])
        fln = keras_model.get_layer(f"{prefix_k}_final_layer_norm")
        assign_ln(
            fln,
            sd[f"{p}.final_layer_norm.weight"],
            sd[f"{p}.final_layer_norm.bias"],
        )

    # Pixel decoder level_embed (separate from transformer module's level_embed).
    pd_level_embed = keras_model.get_layer("pixel_decoder_level_embed")
    pd_level_embed.weight.assign(sd[f"{prefix}.level_embed"])

    # adapter_1, layer_1, mask_projection
    adapter_conv = keras_model.get_layer("pixel_decoder_adapter_1_conv")
    adapter_norm = keras_model.get_layer("pixel_decoder_adapter_1_norm")
    assign_conv(adapter_conv, sd[f"{prefix}.adapter_1.0.weight"])
    assign_ln(
        adapter_norm,
        sd[f"{prefix}.adapter_1.1.weight"],
        sd[f"{prefix}.adapter_1.1.bias"],
    )
    layer1_conv = keras_model.get_layer("pixel_decoder_layer_1_conv")
    layer1_norm = keras_model.get_layer("pixel_decoder_layer_1_norm")
    assign_conv(layer1_conv, sd[f"{prefix}.layer_1.0.weight"])
    assign_ln(
        layer1_norm,
        sd[f"{prefix}.layer_1.1.weight"],
        sd[f"{prefix}.layer_1.1.bias"],
    )
    mask_proj = keras_model.get_layer("pixel_decoder_mask_projection")
    assign_conv(
        mask_proj,
        sd[f"{prefix}.mask_projection.weight"],
        sd[f"{prefix}.mask_projection.bias"],
    )


def transfer_transformer(keras_model, sd, prefix, num_layers, num_heads):
    """Transfer Mask2Former masked-attention decoder weights."""
    queries_features = keras_model.get_layer("transformer_decoder_queries_features")
    queries_features.weight.assign(sd[f"{prefix}.queries_features.weight"])
    queries_embedder = keras_model.get_layer("transformer_decoder_queries_embedder")
    queries_embedder.weight.assign(sd[f"{prefix}.queries_embedder.weight"])

    transformer_level_embed = keras_model.get_layer("transformer_decoder_level_embed")
    transformer_level_embed.weight.assign(sd[f"{prefix}.level_embed.weight"])

    for i in range(num_layers):
        p = f"{prefix}.decoder.layers.{i}"
        prefix_k = f"transformer_decoder_layers_{i}"

        # Self-attention
        sa = keras_model.get_layer(f"{prefix_k}_self_attn")
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            proj_layer = getattr(sa, proj)
            assign_dense(
                proj_layer,
                sd[f"{p}.self_attn.{proj}.weight"],
                sd[f"{p}.self_attn.{proj}.bias"],
            )
        sa_ln = keras_model.get_layer(f"{prefix_k}_self_attn_layer_norm")
        assign_ln(
            sa_ln,
            sd[f"{p}.self_attn_layer_norm.weight"],
            sd[f"{p}.self_attn_layer_norm.bias"],
        )

        # Cross-attention (HF uses fused in_proj_weight = (3*hidden, hidden))
        ca = keras_model.get_layer(f"{prefix_k}_cross_attn")
        ca.in_proj_weight.assign(sd[f"{p}.cross_attn.in_proj_weight"])
        ca.in_proj_bias.assign(sd[f"{p}.cross_attn.in_proj_bias"])
        assign_dense(
            ca.out_proj,
            sd[f"{p}.cross_attn.out_proj.weight"],
            sd[f"{p}.cross_attn.out_proj.bias"],
        )
        ca_ln = keras_model.get_layer(f"{prefix_k}_cross_attn_layer_norm")
        assign_ln(
            ca_ln,
            sd[f"{p}.cross_attn_layer_norm.weight"],
            sd[f"{p}.cross_attn_layer_norm.bias"],
        )

        # FFN
        fc1 = keras_model.get_layer(f"{prefix_k}_fc1")
        fc2 = keras_model.get_layer(f"{prefix_k}_fc2")
        assign_dense(fc1, sd[f"{p}.fc1.weight"], sd[f"{p}.fc1.bias"])
        assign_dense(fc2, sd[f"{p}.fc2.weight"], sd[f"{p}.fc2.bias"])
        fln = keras_model.get_layer(f"{prefix_k}_final_layer_norm")
        assign_ln(
            fln,
            sd[f"{p}.final_layer_norm.weight"],
            sd[f"{p}.final_layer_norm.bias"],
        )

    decoder_ln = keras_model.get_layer("transformer_decoder_layernorm")
    assign_ln(
        decoder_ln,
        sd[f"{prefix}.decoder.layernorm.weight"],
        sd[f"{prefix}.decoder.layernorm.bias"],
    )

    # Mask embedder (3-layer MLP)
    for i in range(3):
        layer = keras_model.get_layer(f"transformer_decoder_mask_embedder_{i}")
        assign_dense(
            layer,
            sd[f"{prefix}.decoder.mask_predictor.mask_embedder.{i}.0.weight"],
            sd[f"{prefix}.decoder.mask_predictor.mask_embedder.{i}.0.bias"],
        )


def transfer_mask2former_weights(keras_model, hf_state_dict):
    """Transfer all Mask2Former weights into a Keras Mask2FormerUniversalSegment."""
    sd = hf_state_dict
    backbone = keras_model.get_layer("backbone")

    backbone_prefix = "model.pixel_level_module.encoder"
    pixel_decoder_prefix = "model.pixel_level_module.decoder"
    transformer_prefix = "model.transformer_module"

    print("Transferring Swin backbone...", flush=True)
    transfer_swin_backbone(backbone, sd, backbone_prefix)

    print("Transferring pixel decoder...", flush=True)
    transfer_pixel_decoder(
        keras_model, sd, pixel_decoder_prefix, keras_model.encoder_layers
    )

    print("Transferring transformer decoder...", flush=True)
    transfer_transformer(
        keras_model,
        sd,
        transformer_prefix,
        keras_model.decoder_layers,
        keras_model.num_heads,
    )

    print("Transferring class_predictor...", flush=True)
    class_pred = keras_model.get_layer("class_predictor")
    assign_dense(class_pred, sd["class_predictor.weight"], sd["class_predictor.bias"])


MASK2FORMER_CONVERSION_CONFIG: List[Dict[str, Any]] = [
    {
        "variant": "mask2former-swin-tiny-coco-instance",
        "hf_id": "facebook/mask2former-swin-tiny-coco-instance",
    },
]


if __name__ == "__main__":
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

        transfer_mask2former_weights(keras_model, sd)

        input_size = keras_model.input_image_shape[0]
        rng = np.random.default_rng(42)
        img_np = rng.integers(0, 255, size=(input_size, input_size, 3), dtype=np.uint8)
        from PIL import Image as PILImage

        image = PILImage.fromarray(img_np)
        hf_proc = Mask2FormerImageProcessor.from_pretrained(hf_id)
        hf_inputs = hf_proc(images=image, return_tensors="pt")

        with torch.no_grad():
            hf_out = hf_model(pixel_values=hf_inputs["pixel_values"])
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
