import re
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from kerasformers.weight_utils.custom_exception import WeightMappingError
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights

WEIGHT_NAME_MAPPING = {
    "embeddings/word_embeddings/embeddings": "embeddings.word_embeddings.weight",
    "embeddings/position_embeddings/embeddings": "embeddings.position_embeddings.weight",
    "embeddings/LayerNorm/gamma": "embeddings.LayerNorm.weight",
    "embeddings/LayerNorm/beta": "embeddings.LayerNorm.bias",
    "vocab_transform/kernel": "vocab_transform.weight",
    "vocab_transform/bias": "vocab_transform.bias",
    "vocab_layer_norm/gamma": "vocab_layer_norm.weight",
    "vocab_layer_norm/beta": "vocab_layer_norm.bias",
    "vocab_projector/kernel": "embeddings.word_embeddings.weight",
    "vocab_projector/bias": "vocab_projector.bias",
    "pre_classifier/kernel": "pre_classifier.weight",
    "pre_classifier/bias": "pre_classifier.bias",
    "classifier/kernel": "classifier.weight",
    "classifier/bias": "classifier.bias",
    "qa_outputs/kernel": "qa_outputs.weight",
    "qa_outputs/bias": "qa_outputs.bias",
}

_OPTIONAL_HEADS = (
    "vocab_transform",
    "vocab_layer_norm",
    "vocab_projector",
    "pre_classifier",
    "classifier",
    "qa_outputs",
)


def hf_name_for(path: str) -> Optional[str]:
    if path in WEIGHT_NAME_MAPPING:
        return WEIGHT_NAME_MAPPING[path]

    m = re.match(
        r"blocks_(\d+)_attention/blocks_\d+_(q_lin|k_lin|v_lin|out_lin)/(kernel|bias)$",
        path,
    )
    if m:
        idx, proj, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"transformer.layer.{idx}.attention.{proj}.{suffix}"

    m = re.match(r"blocks_(\d+)_ffn_(lin1|lin2)/(kernel|bias)$", path)
    if m:
        idx, lin, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"transformer.layer.{idx}.ffn.{lin}.{suffix}"

    m = re.match(r"blocks_(\d+)_(sa_layer_norm|output_layer_norm)/(gamma|beta)$", path)
    if m:
        idx, norm, w = m.groups()
        suffix = "weight" if w == "gamma" else "bias"
        return f"transformer.layer.{idx}.{norm}.{suffix}"

    return None


def normalize_hf_key(key: str) -> str:
    if key.startswith("distilbert."):
        key = key[len("distilbert.") :]
    return key


def transfer_distilbert_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    hf = {normalize_hf_key(k): v for k, v in hf_state_dict.items()}
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        hf_name = hf_name_for(weight.path)
        if hf_name is None:
            continue
        if hf_name not in hf:
            if weight.path.startswith(_OPTIONAL_HEADS):
                continue
            raise WeightMappingError(weight.path, hf_name)
        transfer_weights(weight.path, weight, hf[hf_name])


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    from transformers import DistilBertForMaskedLM
    from transformers import DistilBertModel as HFDistilBertModel

    from kerasformers.models.distilbert import DistilBertMaskedLM, DistilBertModel
    from kerasformers.models.distilbert.config import (
        DISTILBERT_MODEL_CONFIG,
        DISTILBERT_WEIGHT_CONFIG,
    )

    HF_TOKEN = os.environ.get("HF_TOKEN")

    HF_SOURCES = {
        "distilbert_base_uncased": "distilbert-base-uncased",
        "distilbert_base_cased": "distilbert-base-cased",
        "distilbert_base_multilingual_cased": "distilbert-base-multilingual-cased",
    }

    rng = np.random.default_rng(0)

    for variant, meta in DISTILBERT_WEIGHT_CONFIG.items():
        arch = DISTILBERT_MODEL_CONFIG[meta["model"]]
        hf_id = HF_SOURCES[variant]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        ids = rng.integers(0, arch["vocab_size"], (2, 16)).astype("int64")
        mask = np.ones((2, 16), dtype="int64")
        mask[0, 12:] = 0
        k_inputs = {
            "input_ids": ids.astype("int32"),
            "attention_mask": mask.astype("int32"),
        }
        pt = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
        }

        hf_model = HFDistilBertModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        keras_model = DistilBertModel(**arch)
        transfer_distilbert_weights(keras_model, dict(hf_model.state_dict()))
        with torch.no_grad():
            hf_out = hf_model(**pt).last_hidden_state.detach().cpu().numpy()
        seq = keras_model(k_inputs, training=False)["last_hidden_state"]
        seq = seq.detach().cpu().numpy() if hasattr(seq, "detach") else np.asarray(seq)
        d_seq = float(np.abs(hf_out - seq).max())
        print(f"  last_hidden_state max diff: {d_seq:.3e}")
        if d_seq > 1e-3:
            raise ValueError(f"{variant}: DistilBertModel parity failed ({d_seq:.3e})")
        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        hf_mlm = DistilBertForMaskedLM.from_pretrained(hf_id, token=HF_TOKEN).eval()
        keras_mlm = DistilBertMaskedLM(**arch)
        transfer_distilbert_weights(keras_mlm, dict(hf_mlm.state_dict()))
        with torch.no_grad():
            hf_logits = hf_mlm(**pt).logits.detach().cpu().numpy()
        k_logits = keras_mlm(k_inputs, training=False)
        k_logits = (
            k_logits.detach().cpu().numpy()
            if hasattr(k_logits, "detach")
            else np.asarray(k_logits)
        )
        d_mlm = float(np.abs(hf_logits - k_logits).max())
        print(f"  mlm logits max diff: {d_mlm:.3e}")
        if d_mlm > 1e-3:
            raise ValueError(
                f"{variant}: DistilBertMaskedLM parity failed ({d_mlm:.3e})"
            )
        out_path = f"{variant}_mlm.weights.h5"
        keras_mlm.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del hf_model, hf_mlm, keras_model, keras_mlm
        keras.backend.clear_session()
        gc.collect()
