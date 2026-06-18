from tqdm import tqdm

from kerasformers.conversion.weight_transfer_util import transfer_weights


def transfer_fused_attention(keras_attn, sd, prefix):
    # torch nn.MultiheadAttention: fused in_proj + out_proj.
    keras_attn.in_proj_weight.assign(sd[f"{prefix}.in_proj_weight"])
    keras_attn.in_proj_bias.assign(sd[f"{prefix}.in_proj_bias"])
    transfer_weights(
        "kernel", keras_attn.out_proj.weights[0], sd[f"{prefix}.out_proj.weight"]
    )
    transfer_weights(
        "bias", keras_attn.out_proj.weights[1], sd[f"{prefix}.out_proj.bias"]
    )


def transfer_dense(keras_layer, sd, prefix):
    transfer_weights("kernel", keras_layer.weights[0], sd[f"{prefix}.weight"])
    transfer_weights("bias", keras_layer.weights[1], sd[f"{prefix}.bias"])


def transfer_layernorm(keras_layer, sd, prefix):
    transfer_weights("gamma", keras_layer.weights[0], sd[f"{prefix}.weight"])
    transfer_weights("beta", keras_layer.weights[1], sd[f"{prefix}.bias"])


def transfer_oneformer_weights(keras_model, hf_state_dict):
    """Transfer a OneFormer state dict into the Keras OneFormer model.

    Walks the Swin backbone, the MSDeformAttn pixel decoder, the task MLP,
    the query transformer, the masked-attention decoder, and the class / mask
    heads. The training-only ``text_mapper`` (absent from the released
    ``is_training: false`` checkpoints) is ignored.
    """
    sd = hf_state_dict
    backbone = keras_model.get_layer("backbone")

    backbone_prefix = "model.pixel_level_module.encoder"
    pixel_decoder_prefix = "model.pixel_level_module.decoder"
    transformer_prefix = "model.transformer_module"
    decoder_prefix = f"{transformer_prefix}.decoder"

    print("Transferring Swin backbone...", flush=True)
    proj = backbone.patch_embeddings.projection
    transfer_weights(
        "conv_kernel",
        proj.weights[0],
        sd[f"{backbone_prefix}.embeddings.patch_embeddings.projection.weight"],
    )
    transfer_weights(
        "bias",
        proj.weights[1],
        sd[f"{backbone_prefix}.embeddings.patch_embeddings.projection.bias"],
    )
    transfer_layernorm(
        backbone.embeddings_norm, sd, f"{backbone_prefix}.embeddings.norm"
    )

    for stage_idx, stage in enumerate(backbone.stages):
        for block_idx, block in enumerate(stage.blocks):
            p = f"{backbone_prefix}.encoder.layers.{stage_idx}.blocks.{block_idx}"
            transfer_layernorm(block.layernorm_before, sd, f"{p}.layernorm_before")
            attn = block.attention.self_attn
            for name in ("query", "key", "value"):
                transfer_dense(getattr(attn, name), sd, f"{p}.attention.self.{name}")
            attn.relative_position_bias_table.assign(
                sd[f"{p}.attention.self.relative_position_bias_table"]
            )
            transfer_dense(
                block.attention.output_dense, sd, f"{p}.attention.output.dense"
            )
            transfer_layernorm(block.layernorm_after, sd, f"{p}.layernorm_after")
            transfer_dense(block.intermediate_dense, sd, f"{p}.intermediate.dense")
            transfer_dense(block.output_dense, sd, f"{p}.output.dense")

        if stage.downsample is not None:
            ds = f"{backbone_prefix}.encoder.layers.{stage_idx}.downsample"
            transfer_weights(
                "kernel",
                stage.downsample.reduction.weights[0],
                sd[f"{ds}.reduction.weight"],
            )
            transfer_layernorm(stage.downsample.norm, sd, f"{ds}.norm")

    for i in range(len(backbone.hidden_states_norms)):
        transfer_layernorm(
            backbone.hidden_states_norms[i],
            sd,
            f"{backbone_prefix}.hidden_states_norms.stage{i + 1}",
        )

    print("Transferring pixel decoder...", flush=True)
    for i in range(3):
        p_proj = f"{pixel_decoder_prefix}.input_projections.{i}"
        conv = keras_model.get_layer(f"pixel_decoder_input_projections_{i}_conv")
        transfer_weights("conv_kernel", conv.weights[0], sd[f"{p_proj}.0.weight"])
        transfer_weights("bias", conv.weights[1], sd[f"{p_proj}.0.bias"])
        nrm = keras_model.get_layer(f"pixel_decoder_input_projections_{i}_norm")
        transfer_weights("gamma", nrm.weights[0], sd[f"{p_proj}.1.weight"])
        transfer_weights("beta", nrm.weights[1], sd[f"{p_proj}.1.bias"])

    for i in tqdm(
        range(keras_model.encoder_num_layers), desc="Transferring encoder layers"
    ):
        p = f"{pixel_decoder_prefix}.encoder.layers.{i}"
        prefix_k = f"pixel_decoder_encoder_layers_{i}"
        attn = keras_model.get_layer(f"{prefix_k}_self_attn")
        for name in (
            "sampling_offsets",
            "attention_weights",
            "value_proj",
            "output_proj",
        ):
            transfer_dense(getattr(attn, name), sd, f"{p}.self_attn.{name}")
        transfer_layernorm(
            keras_model.get_layer(f"{prefix_k}_self_attn_layer_norm"),
            sd,
            f"{p}.self_attn_layer_norm",
        )
        transfer_dense(keras_model.get_layer(f"{prefix_k}_fc1"), sd, f"{p}.fc1")
        transfer_dense(keras_model.get_layer(f"{prefix_k}_fc2"), sd, f"{p}.fc2")
        transfer_layernorm(
            keras_model.get_layer(f"{prefix_k}_final_layer_norm"),
            sd,
            f"{p}.final_layer_norm",
        )

    keras_model.get_layer("pixel_decoder_level_embed").weight.assign(
        sd[f"{pixel_decoder_prefix}.level_embed"]
    )
    adapter_conv = keras_model.get_layer("pixel_decoder_adapter_1_conv")
    transfer_weights(
        "conv_kernel",
        adapter_conv.weights[0],
        sd[f"{pixel_decoder_prefix}.adapter_1.0.weight"],
    )
    adapter_norm = keras_model.get_layer("pixel_decoder_adapter_1_norm")
    transfer_weights(
        "gamma",
        adapter_norm.weights[0],
        sd[f"{pixel_decoder_prefix}.adapter_1.1.weight"],
    )
    transfer_weights(
        "beta", adapter_norm.weights[1], sd[f"{pixel_decoder_prefix}.adapter_1.1.bias"]
    )
    layer1_conv = keras_model.get_layer("pixel_decoder_layer_1_conv")
    transfer_weights(
        "conv_kernel",
        layer1_conv.weights[0],
        sd[f"{pixel_decoder_prefix}.layer_1.0.weight"],
    )
    layer1_norm = keras_model.get_layer("pixel_decoder_layer_1_norm")
    transfer_weights(
        "gamma", layer1_norm.weights[0], sd[f"{pixel_decoder_prefix}.layer_1.1.weight"]
    )
    transfer_weights(
        "beta", layer1_norm.weights[1], sd[f"{pixel_decoder_prefix}.layer_1.1.bias"]
    )
    mask_proj = keras_model.get_layer("pixel_decoder_mask_projection")
    transfer_weights(
        "conv_kernel",
        mask_proj.weights[0],
        sd[f"{pixel_decoder_prefix}.mask_projection.weight"],
    )
    transfer_weights(
        "bias", mask_proj.weights[1], sd[f"{pixel_decoder_prefix}.mask_projection.bias"]
    )

    print("Transferring task encoder + query transformer...", flush=True)
    for i in range(2):
        transfer_dense(
            keras_model.get_layer(f"task_encoder_task_mlp_{i}"),
            sd,
            f"model.task_encoder.task_mlp.layers.{i}.0",
        )

    keras_model.get_layer("transformer_decoder_queries_embedder").weight.assign(
        sd[f"{transformer_prefix}.queries_embedder.weight"]
    )
    keras_model.get_layer("transformer_decoder_level_embed").weight.assign(
        sd[f"{transformer_prefix}.level_embed.weight"]
    )
    qip = keras_model.get_layer("transformer_decoder_query_input_projection")
    transfer_weights(
        "conv_kernel",
        qip.weights[0],
        sd[f"{decoder_prefix}.query_input_projection.weight"],
    )
    transfer_weights(
        "bias", qip.weights[1], sd[f"{decoder_prefix}.query_input_projection.bias"]
    )

    for i in range(keras_model.query_dec_layers):
        p = f"{decoder_prefix}.query_transformer.decoder.layers.{i}"
        prefix_k = f"transformer_decoder_query_transformer_layers_{i}"
        transfer_fused_attention(
            keras_model.get_layer(f"{prefix_k}_self_attn"), sd, f"{p}.self_attn"
        )
        transfer_fused_attention(
            keras_model.get_layer(f"{prefix_k}_cross_attn"), sd, f"{p}.multihead_attn"
        )
        transfer_dense(keras_model.get_layer(f"{prefix_k}_linear1"), sd, f"{p}.linear1")
        transfer_dense(keras_model.get_layer(f"{prefix_k}_linear2"), sd, f"{p}.linear2")
        for n in (1, 2, 3):
            transfer_layernorm(
                keras_model.get_layer(f"{prefix_k}_norm{n}"), sd, f"{p}.norm{n}"
            )
    transfer_layernorm(
        keras_model.get_layer("transformer_decoder_query_transformer_norm"),
        sd,
        f"{decoder_prefix}.query_transformer.decoder.norm",
    )

    print("Transferring transformer decoder...", flush=True)
    for i in tqdm(
        range(keras_model.decoder_num_layers), desc="Transferring decoder layers"
    ):
        p = f"{decoder_prefix}.layers.{i}"
        prefix_k = f"transformer_decoder_layers_{i}"
        transfer_fused_attention(
            keras_model.get_layer(f"{prefix_k}_cross_attn"),
            sd,
            f"{p}.cross_attn.multihead_attn",
        )
        transfer_layernorm(
            keras_model.get_layer(f"{prefix_k}_cross_attn_norm"),
            sd,
            f"{p}.cross_attn.norm",
        )
        sa = keras_model.get_layer(f"{prefix_k}_self_attn")
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            transfer_dense(getattr(sa, proj), sd, f"{p}.self_attn.self_attn.{proj}")
        transfer_layernorm(
            keras_model.get_layer(f"{prefix_k}_self_attn_norm"),
            sd,
            f"{p}.self_attn.norm",
        )
        transfer_dense(
            keras_model.get_layer(f"{prefix_k}_linear1"), sd, f"{p}.ffn.linear1"
        )
        transfer_dense(
            keras_model.get_layer(f"{prefix_k}_linear2"), sd, f"{p}.ffn.linear2"
        )
        transfer_layernorm(
            keras_model.get_layer(f"{prefix_k}_ffn_norm"), sd, f"{p}.ffn.norm"
        )

    transfer_layernorm(
        keras_model.get_layer("transformer_decoder_norm"),
        sd,
        f"{decoder_prefix}.decoder_norm",
    )
    transfer_dense(
        keras_model.get_layer("transformer_decoder_class_embed"),
        sd,
        f"{decoder_prefix}.class_embed",
    )
    for i in range(3):
        transfer_dense(
            keras_model.get_layer(f"transformer_decoder_mask_embed_{i}"),
            sd,
            f"{decoder_prefix}.mask_embed.layers.{i}.0",
        )


if __name__ == "__main__":
    # Release-weights driver: convert every variant from its HF checkpoint,
    # check HF-vs-Keras parity, and save .weights.h5 (or a sharded .weights.json
    # when >2 GB). Run with KERAS_BACKEND=torch. Verify the HF class name and the
    # output key (`class_queries_logits`) against your transformers version.
    import gc
    import os

    import keras
    import numpy as np
    import torch
    import transformers
    from PIL import Image

    from kerasformers.models.oneformer import (
        OneFormerProcessor,
        OneFormerUniversalSegment,
    )
    from kerasformers.models.oneformer.config import ONEFORMER_WEIGHTS_URLS

    HF_SOURCES = {
        "oneformer_ade20k_swin_tiny": "shi-labs/oneformer_ade20k_swin_tiny",
        "oneformer_ade20k_swin_large": "shi-labs/oneformer_ade20k_swin_large",
        "oneformer_coco_swin_large": "shi-labs/oneformer_coco_swin_large",
        "oneformer_cityscapes_swin_large": "shi-labs/oneformer_cityscapes_swin_large",
    }
    MAX_SHARD_GB = 1.7  # GitHub caps a single release asset at 2 GB
    rng = np.random.default_rng(0)

    def cosine(a, b):
        a, b = a.astype("float64").ravel(), b.astype("float64").ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    for variant, meta in ONEFORMER_WEIGHTS_URLS.items():
        hf_id = HF_SOURCES[variant]
        out_path = os.path.basename(meta["url"])
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        model = OneFormerUniversalSegment.from_weights("hf:" + hf_id)

        # HF-vs-Keras parity on identical inputs from the kerasformers processor.
        img = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype="uint8"))
        proc = OneFormerProcessor.from_weights("hf:" + hf_id)
        kin = proc(images=img, task="semantic")
        k_logits = np.asarray(
            keras.ops.convert_to_numpy(model(kin)["class_queries_logits"])
        )
        hf_model = transformers.OneFormerForUniversalSegmentation.from_pretrained(
            hf_id
        ).eval()
        pv = np.transpose(
            np.asarray(keras.ops.convert_to_numpy(kin["pixel_values"])), (0, 3, 1, 2)
        )
        ti = np.asarray(keras.ops.convert_to_numpy(kin["task_inputs"])).astype("int64")
        with torch.no_grad():
            hf_out = hf_model(
                pixel_values=torch.from_numpy(pv), task_inputs=torch.from_numpy(ti)
            )
        cos = cosine(k_logits, hf_out.class_queries_logits.numpy())
        print(f"  class_queries_logits cosine: {cos:.6f}")
        if cos < 0.99:
            raise ValueError(f"{variant}: OneFormer parity failed (cos={cos:.4f})")

        n_bytes = sum(int(np.prod(w.shape)) * 4 for w in model.weights)
        if out_path.endswith(".json"):
            model.save_weights(out_path, max_shard_size=MAX_SHARD_GB)
        elif n_bytes > 2 * 1024**3:
            raise ValueError(
                f"{variant} is {n_bytes / 1024**3:.2f} GB (> 2 GB); set its config "
                f"URL extension to .weights.json so it shards."
            )
        else:
            model.save_weights(out_path)
        print(f"  Saved -> {out_path}  ({n_bytes / 1024**3:.2f} GB)")

        del hf_model, model
        keras.backend.clear_session()
        gc.collect()
