import re

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

SWIN_PREFIX = "model.backbone.conv_encoder.model.swin"
BB_PREFIX = "model.backbone.conv_encoder.model"


def keras_to_hf(name):
    """Map a Keras weight path (dots) to its HF Grounding DINO key."""
    # --- Swin backbone ---
    if name.startswith("backbone."):
        n = name[len("backbone.") :]
        if n.startswith("patch_embeddings_projection"):
            return f"{SWIN_PREFIX}.embeddings.patch_embeddings.projection." + (
                "weight" if n.endswith("kernel") else "bias"
            )
        if n.startswith("embed_norm"):
            return f"{SWIN_PREFIX}.embeddings.norm." + (
                "weight" if n.endswith("gamma") else "bias"
            )
        m = re.match(r"hidden_states_norms_stage(\d+)\.(gamma|beta)", n)
        if m:
            return f"{BB_PREFIX}.hidden_states_norms.stage{m.group(1)}." + (
                "weight" if m.group(2) == "gamma" else "bias"
            )
        m = re.match(r"stage_(\d+)\.(.*)", n)
        if m:
            stage, rest = m.group(1), m.group(2)
            rest = rest.replace("blocks_", "blocks.")
            rest = rest.replace(
                "attention.relative_position_bias_table",
                "attention.relative_position_bias.relative_position_bias_table",
            )
            rest = rest.replace("mlp_fc1", "mlp.fc1").replace("mlp_fc2", "mlp.fc2")
            rest = (
                rest.replace("kernel", "weight")
                .replace("gamma", "weight")
                .replace("beta", "bias")
            )
            return f"{SWIN_PREFIX}.encoder.layers.{stage}.{rest}"
    # --- text backbone (BERT) ---
    if name.startswith("text_backbone."):
        n = name[len("text_backbone.") :]
        if n.startswith("embeddings."):
            tail = n[len("embeddings.") :]
            if tail.endswith(".embeddings"):  # keras Embedding weight
                return f"model.text_backbone.embeddings.{tail[: -len('.embeddings')]}.weight"
            tail = tail.replace("gamma", "weight").replace("beta", "bias")
            return f"model.text_backbone.embeddings.{tail}"
        m = re.match(r"layer_(\d+)\.(.*)", n)
        if m:
            i, rest = m.group(1), m.group(2)
            mapping = {
                "query": "attention.self.query",
                "key": "attention.self.key",
                "value": "attention.self.value",
                "attn_output": "attention.output.dense",
                "attn_norm": "attention.output.LayerNorm",
                "intermediate": "intermediate.dense",
                "output_dense": "output.dense",
                "output_norm": "output.LayerNorm",
            }
            for k, v in mapping.items():
                if rest.startswith(k + "."):
                    rest = v + rest[len(k) :]
                    break
            rest = (
                rest.replace("kernel", "weight")
                .replace("gamma", "weight")
                .replace("beta", "bias")
            )
            return f"model.text_backbone.encoder.layer.{i}.{rest}"
    if name.startswith("text_projection."):
        return "model.text_projection." + (
            "weight" if name.endswith("kernel") else "bias"
        )
    # --- input projections ---
    m = re.match(r"input_proj_(\d+)_conv\.(kernel|bias)", name)
    if m:
        return f"model.input_proj_vision.{m.group(1)}.0." + (
            "weight" if m.group(2) == "kernel" else "bias"
        )
    m = re.match(r"input_proj_(\d+)_norm\.(gamma|beta)", name)
    if m:
        return f"model.input_proj_vision.{m.group(1)}.1." + (
            "weight" if m.group(2) == "gamma" else "bias"
        )
    if name == "level_embed":
        return "model.level_embed"
    # --- encoder layers ---
    m = re.match(r"encoder_layer_(\d+)\.(.*)", name)
    if m:
        rest = _common(m.group(2))
        return f"model.encoder.layers.{m.group(1)}.{rest}"
    # --- two-stage ---
    if name.startswith("enc_output_norm."):
        return "model.enc_output_norm." + (
            "weight" if name.endswith("gamma") else "bias"
        )
    if name.startswith("enc_output."):
        return "model.enc_output." + ("weight" if name.endswith("kernel") else "bias")
    m = re.match(r"encoder_output_bbox_embed\.layers_(\d+)\.(kernel|bias)", name)
    if m:
        return f"model.encoder_output_bbox_embed.layers.{m.group(1)}." + (
            "weight" if m.group(2) == "kernel" else "bias"
        )
    if name.startswith("query_position_embeddings.embeddings"):
        return "model.query_position_embeddings.weight"
    # --- decoder ---
    if name.startswith("decoder_norm."):
        return "model.decoder.layer_norm." + (
            "weight" if name.endswith("gamma") else "bias"
        )
    m = re.match(r"reference_points_head\.layers_(\d+)\.(kernel|bias)", name)
    if m:
        return f"model.decoder.reference_points_head.layers.{m.group(1)}." + (
            "weight" if m.group(2) == "kernel" else "bias"
        )
    m = re.match(r"decoder_layer_(\d+)\.(.*)", name)
    if m:
        rest = _common(m.group(2))
        return f"model.decoder.layers.{m.group(1)}.{rest}"
    # --- detection heads ---
    m = re.match(r"bbox_embed_(\d+)\.layers_(\d+)\.(kernel|bias)", name)
    if m:
        return f"bbox_embed.{m.group(1)}.layers.{m.group(2)}." + (
            "weight" if m.group(3) == "kernel" else "bias"
        )
    raise WeightMappingError(name, name)


def _common(rest):
    """Shared encoder/decoder sublayer name fixups (Dense/LayerNorm/params)."""
    rest = rest.replace("self_attn_layer_norm", "self_attn_layer_norm")
    rest = (
        rest.replace("kernel", "weight")
        .replace("gamma", "weight")
        .replace("beta", "bias")
    )
    return rest


def transfer_grounding_dino_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        keras_model(
            {
                "input_ids": np.array(
                    [[101, 102, 1012, 1029, 102, 102]], dtype="int64"
                ),
                "attention_mask": np.ones((1, 6), dtype="int64"),
                "pixel_values": np.zeros((1, 224, 224, 3), dtype="float32"),
            }
        )
    state = hf_state_dict
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        kname = weight.path.split("/", 1)[1].replace("/", ".")
        hf = keras_to_hf(kname)
        if hf not in state and re.match(r"bbox_embed\.\d+\.", hf):
            # decoder_bbox_embed_share=True -> all share bbox_embed.0
            hf = re.sub(r"bbox_embed\.\d+\.", "bbox_embed.0.", hf)
        if hf not in state and hf.startswith("model.") and hf[len("model.") :] in state:
            # base GroundingDinoModel state dicts omit the `model.` prefix.
            hf = hf[len("model.") :]
        if hf not in state:
            raise WeightMappingError(weight.path, hf)
        value = state[hf]
        if kname.endswith("patch_embeddings_projection.kernel") or re.search(
            r"input_proj_\d+_conv\.kernel", kname
        ):
            weight.assign(np.transpose(np.asarray(value), (2, 3, 1, 0)))
        elif (
            kname == "level_embed"
            or kname.endswith("relative_position_bias_table")
            or kname.endswith("fusion_layer.vision_param")
            or kname.endswith("fusion_layer.text_param")
        ):
            weight.assign(np.asarray(value))
        else:
            transfer_weights(weight.path, weight, value)
