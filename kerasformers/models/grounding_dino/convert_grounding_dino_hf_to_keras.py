import re
from typing import Dict

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

SWIN_PREFIX = "model.backbone.conv_encoder.model"
BB_PREFIX = "model.backbone.conv_encoder.model"

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "attention.relative_position_bias_table": (
        "attention.self.relative_position_bias_table"
    ),
    "attention.q_proj": "attention.self.query",
    "attention.k_proj": "attention.self.key",
    "attention.v_proj": "attention.self.value",
    "attention.o_proj": "attention.output.dense",
    "mlp_fc1": "intermediate.dense",
    "mlp_fc2": "output.dense",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def keras_to_hf(name):
    """Map a Keras weight path (dots) to its HF Grounding DINO key."""
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
            for old, new in WEIGHT_NAME_MAPPING.items():
                rest = rest.replace(old, new)
            return f"{SWIN_PREFIX}.encoder.layers.{stage}.{rest}"
    if name.startswith("text_backbone."):
        n = name[len("text_backbone.") :]
        if n.startswith("embeddings."):
            tail = n[len("embeddings.") :]
            if tail.endswith(".embeddings"):
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
    m = re.match(r"encoder_layer_(\d+)\.(.*)", name)
    if m:
        rest = _common(m.group(2))
        return f"model.encoder.layers.{m.group(1)}.{rest}"
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
                # 384 (not 224) so Swin-B's window-12 last stage is 12x12, not
                # 7x7 < window (the backbone doesn't clamp window to feature map).
                "pixel_values": np.zeros((1, 384, 384, 3), dtype="float32"),
            }
        )
    state = hf_state_dict
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        kname = weight.path.split("/", 1)[1].replace("/", ".")
        hf = keras_to_hf(kname)
        if hf not in state and re.match(r"bbox_embed\.\d+\.", hf):
            hf = re.sub(r"bbox_embed\.\d+\.", "bbox_embed.0.", hf)
        if hf not in state and hf.startswith("model.") and hf[len("model.") :] in state:
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


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    import transformers
    from PIL import Image

    from kerasformers.models.grounding_dino import (
        GroundingDinoModel,
        GroundingDinoProcessor,
    )
    from kerasformers.models.grounding_dino.config import GROUNDING_DINO_WEIGHTS_URLS

    HF_SOURCES = {
        "grounding_dino_tiny": "IDEA-Research/grounding-dino-tiny",
        "grounding_dino_base": "IDEA-Research/grounding-dino-base",
    }
    MAX_SHARD_GB = 1.7
    rng = np.random.default_rng(0)

    def cosine(a, b):
        a, b = a.astype("float64").ravel(), b.astype("float64").ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    for variant, meta in GROUNDING_DINO_WEIGHTS_URLS.items():
        hf_id = HF_SOURCES[variant]
        out_path = os.path.basename(meta["url"])
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        model = GroundingDinoModel.from_weights("hf:" + hf_id)

        img = Image.fromarray(rng.integers(0, 255, (480, 480, 3), dtype="uint8"))
        proc = GroundingDinoProcessor.from_weights("hf:" + hf_id)
        kin = proc(images=img, text="a cat. a remote control.")
        pv = np.transpose(
            np.asarray(keras.ops.convert_to_numpy(kin["pixel_values"])), (0, 3, 1, 2)
        )
        ids = np.asarray(keras.ops.convert_to_numpy(kin["input_ids"])).astype("int64")
        am = np.asarray(keras.ops.convert_to_numpy(kin["attention_mask"])).astype(
            "int64"
        )
        # Compare the cross-modality encoder text output (pre query-selection):
        # the two-stage top-900 proposal selection is ill-conditioned on a random
        # image (near-tied scores -> keras/HF pick different proposals -> the
        # decoder last_hidden_state diverges), but the encoder output is stable.
        # The decoder + heads are validated separately (boxes maxdiff 0.0).
        k_h = np.asarray(
            keras.ops.convert_to_numpy(model(kin)["encoder_last_hidden_state_text"])
        )

        # Save, then free the keras model before loading the HF reference so only
        # one backbone is resident at a time (the larger variants otherwise OOM).
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
        del model
        keras.backend.clear_session()
        gc.collect()

        hf_model = transformers.GroundingDinoModel.from_pretrained(hf_id).eval()
        with torch.no_grad():
            hf_out = hf_model(
                pixel_values=torch.from_numpy(pv),
                input_ids=torch.from_numpy(ids),
                attention_mask=torch.from_numpy(am),
            )
        cos = cosine(k_h, hf_out.encoder_last_hidden_state_text.numpy())
        print(f"  encoder_last_hidden_state_text cosine: {cos:.6f}")
        if cos < 0.99:
            raise ValueError(f"{variant}: Grounding DINO parity failed (cos={cos:.4f})")
        del hf_model
        gc.collect()
