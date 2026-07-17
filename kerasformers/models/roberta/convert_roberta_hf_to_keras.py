import re
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "embeddings/word_embeddings/embeddings": "embeddings.word_embeddings.weight",
    "embeddings/position_embeddings/embeddings": "embeddings.position_embeddings.weight",
    "embeddings/token_type_embeddings/embeddings": "embeddings.token_type_embeddings.weight",
    "embeddings/LayerNorm/gamma": "embeddings.LayerNorm.weight",
    "embeddings/LayerNorm/beta": "embeddings.LayerNorm.bias",
    "attention_output_dense": "attention.output.dense",
    "intermediate_dense": "intermediate.dense",
    "output_dense": "output.dense",
    "attention_output_layernorm": "attention.output.LayerNorm",
    "output_layernorm": "output.LayerNorm",
    "pooler_dense/kernel": "pooler.dense.weight",
    "pooler_dense/bias": "pooler.dense.bias",
    "lm_head_dense/kernel": "lm_head.dense.weight",
    "lm_head_dense/bias": "lm_head.dense.bias",
    "lm_head_layernorm/gamma": "lm_head.layer_norm.weight",
    "lm_head_layernorm/beta": "lm_head.layer_norm.bias",
    # The MLM decoder kernel is tied to the input word embeddings; HF strips the
    # tied `lm_head.decoder.weight` from safetensors, so map to the embedding
    # table (transfer_weights transposes it into the Dense kernel). Mapping to
    # the tied key instead would be silently skipped (lm_head is optional) and
    # leave a random decoder. Mirrors the BERT / DeBERTa converters.
    "lm_head_decoder/kernel": "embeddings.word_embeddings.weight",
    "lm_head_decoder/bias": "lm_head.bias",
    "classifier_dense/kernel": "classifier.dense.weight",
    "classifier_dense/bias": "classifier.dense.bias",
    "classifier_out_proj/kernel": "classifier.out_proj.weight",
    "classifier_out_proj/bias": "classifier.out_proj.bias",
    "classifier/kernel": "classifier.weight",
    "classifier/bias": "classifier.bias",
    "qa_outputs/kernel": "qa_outputs.weight",
    "qa_outputs/bias": "qa_outputs.bias",
}

_OPTIONAL_WEIGHTS = ("classifier", "qa_outputs", "lm_head", "pooler_dense")


def hf_name_for(path: str) -> Optional[str]:
    if path in WEIGHT_NAME_MAPPING:
        return WEIGHT_NAME_MAPPING[path]

    m = re.match(
        r"blocks_(\d+)_attention_self/blocks_\d+_(query|key|value)/(kernel|bias)$",
        path,
    )
    if m:
        idx, proj, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"encoder.layer.{idx}.attention.self.{proj}.{suffix}"

    m = re.match(
        r"blocks_(\d+)_(attention_output_dense|intermediate_dense|output_dense)/(kernel|bias)$",
        path,
    )
    if m:
        idx, layer, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"encoder.layer.{idx}.{WEIGHT_NAME_MAPPING[layer]}.{suffix}"

    m = re.match(
        r"blocks_(\d+)_(attention_output_layernorm|output_layernorm)/(gamma|beta)$",
        path,
    )
    if m:
        idx, layer, w = m.groups()
        suffix = "weight" if w == "gamma" else "bias"
        return f"encoder.layer.{idx}.{WEIGHT_NAME_MAPPING[layer]}.{suffix}"

    return None


def normalize_hf_key(key: str) -> str:
    if key.startswith("roberta."):
        key = key[len("roberta.") :]
    return key.replace("LayerNorm.gamma", "LayerNorm.weight").replace(
        "LayerNorm.beta", "LayerNorm.bias"
    )


def transfer_roberta_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    hf = {normalize_hf_key(k): v for k, v in hf_state_dict.items()}
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        hf_name = hf_name_for(weight.path)
        if hf_name is None:
            continue
        if hf_name not in hf:
            if weight.path.startswith(_OPTIONAL_WEIGHTS):
                continue
            raise WeightMappingError(weight.path, hf_name)
        transfer_weights(weight.path, weight, hf[hf_name])


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    from transformers import RobertaForMaskedLM
    from transformers import RobertaModel as HFRobertaModel

    from kerasformers.models.roberta import RobertaMaskedLM, RobertaModel
    from kerasformers.models.roberta.roberta_config import (
        ROBERTA_MODEL_CONFIG,
        ROBERTA_WEIGHTS_URLS,
    )

    HF_TOKEN = os.environ.get("HF_TOKEN")

    HF_SOURCES = {
        "roberta_base": "FacebookAI/roberta-base",
        "roberta_large": "FacebookAI/roberta-large",
    }

    rng = np.random.default_rng(0)

    for variant, meta in ROBERTA_WEIGHTS_URLS.items():
        arch = ROBERTA_MODEL_CONFIG[meta["model"]]
        hf_id = HF_SOURCES[variant]
        pad = arch["pad_token_id"]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        ids = rng.integers(3, arch["vocab_size"], (2, 16)).astype("int64")
        ids[:, 0] = 0
        ids[0, 12:] = pad
        mask = np.ones((2, 16), dtype="int64")
        mask[0, 12:] = 0
        types = np.zeros((2, 16), dtype="int64")
        k_inputs = {
            "input_ids": ids.astype("int32"),
            "attention_mask": mask.astype("int32"),
            "token_type_ids": types.astype("int32"),
        }
        pt = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
        }

        hf_model = HFRobertaModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        keras_model = RobertaModel(**arch)
        transfer_roberta_weights(keras_model, dict(hf_model.state_dict()))
        with torch.no_grad():
            hf_out = hf_model(**pt)
        k_out = keras_model(k_inputs, training=False)
        seq = k_out["last_hidden_state"].detach().cpu().numpy()
        pool = k_out["pooler_output"].detach().cpu().numpy()
        d_seq = float(
            np.abs(hf_out.last_hidden_state.detach().cpu().numpy() - seq).max()
        )
        d_pool = float(np.abs(hf_out.pooler_output.detach().cpu().numpy() - pool).max())
        print(
            f"  last_hidden_state max diff: {d_seq:.3e}   pooler max diff: {d_pool:.3e}"
        )
        if max(d_seq, d_pool) > 1e-3:
            raise ValueError(f"{variant}: RobertaModel parity failed")
        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        hf_mlm = RobertaForMaskedLM.from_pretrained(hf_id, token=HF_TOKEN).eval()
        keras_mlm = RobertaMaskedLM(**arch)
        # Simulate the safetensors / `hf:` path: the tied MLM decoder kernel is
        # stripped from safetensors, so drop it here too: the converter must
        # reconstruct it from the word embeddings, not depend on the tied key.
        mlm_sd = dict(hf_mlm.state_dict())
        mlm_sd.pop("lm_head.decoder.weight", None)
        transfer_roberta_weights(keras_mlm, mlm_sd)
        with torch.no_grad():
            hf_logits = hf_mlm(**pt).logits
        k_logits = keras_mlm(k_inputs, training=False)
        mlm = k_logits.detach().cpu().numpy()
        d_mlm = float(np.abs(hf_logits.detach().cpu().numpy() - mlm).max())
        print(f"  mlm logits max diff: {d_mlm:.3e}")
        if d_mlm > 1e-3:
            raise ValueError(f"{variant}: RobertaMaskedLM parity failed")
        out_path = f"{variant}_mlm.weights.h5"
        keras_mlm.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del hf_model, hf_mlm, keras_model, keras_mlm
        keras.backend.clear_session()
        gc.collect()
