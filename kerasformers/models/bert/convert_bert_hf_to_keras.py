import re
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from kerasformers.weight_utils.custom_exception import WeightMappingError
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights

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
    "mlm_transform_dense/kernel": "cls.predictions.transform.dense.weight",
    "mlm_transform_dense/bias": "cls.predictions.transform.dense.bias",
    "mlm_transform_layernorm/gamma": "cls.predictions.transform.LayerNorm.weight",
    "mlm_transform_layernorm/beta": "cls.predictions.transform.LayerNorm.bias",
    "mlm_decoder/kernel": "cls.predictions.decoder.weight",
    "mlm_decoder/bias": "cls.predictions.bias",
    # Task heads (present only in fine-tuned sequence/token-classification repos).
    "classifier/kernel": "classifier.weight",
    "classifier/bias": "classifier.bias",
}


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


def transfer_bert_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    hf = {
        (k[len("bert.") :] if k.startswith("bert.") else k): v
        for k, v in hf_state_dict.items()
    }
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        hf_name = hf_name_for(weight.path)
        if hf_name is None:
            continue
        if hf_name not in hf:
            # The task head exists only in fine-tuned repos; leave it randomly
            # initialized when loading a backbone-only / base checkpoint.
            if weight.path.startswith("classifier"):
                continue
            raise WeightMappingError(weight.path, hf_name)
        transfer_weights(weight.path, weight, hf[hf_name])


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    from transformers import BertForMaskedLM
    from transformers import BertModel as HFBertModel

    from kerasformers.models.bert import BertMaskedLM, BertModel
    from kerasformers.models.bert.config import BERT_MODEL_CONFIG, BERT_WEIGHT_CONFIG

    OUT_DIR = "C:/Users/gites/Desktop/code/v1_weights"
    os.makedirs(OUT_DIR, exist_ok=True)
    HF_TOKEN = os.environ.get("HF_TOKEN")

    def max_diff(a, b):
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        return float(np.abs(a - b).max())

    rng = np.random.default_rng(0)

    for variant, meta in BERT_WEIGHT_CONFIG.items():
        arch = BERT_MODEL_CONFIG[meta["model"]]
        hf_id = meta["hf_id"]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        ids = rng.integers(0, arch["vocab_size"], (2, 16)).astype("int64")
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
            "token_type_ids": torch.from_numpy(types),
        }

        # --- BertModel (backbone + pooler) ---
        hf_model = HFBertModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        keras_model = BertModel(**arch)
        transfer_bert_weights(keras_model, dict(hf_model.state_dict()))
        with torch.no_grad():
            hf_out = hf_model(**pt)
        k_out = keras_model(k_inputs, training=False)
        d_seq = max_diff(
            hf_out.last_hidden_state, np.asarray(k_out["last_hidden_state"])
        )
        d_pool = max_diff(hf_out.pooler_output, np.asarray(k_out["pooler_output"]))
        print(
            f"  last_hidden_state max diff: {d_seq:.3e}   pooler max diff: {d_pool:.3e}"
        )
        if max(d_seq, d_pool) > 1e-4:
            raise ValueError(f"{variant}: BertModel parity failed")
        path = f"{OUT_DIR}/{variant}.weights.h5"
        keras_model.save_weights(path)
        print(f"  saved -> {path}")

        # --- BertMaskedLM (backbone + MLM head) ---
        hf_mlm = BertForMaskedLM.from_pretrained(hf_id, token=HF_TOKEN).eval()
        keras_mlm = BertMaskedLM(**arch)
        transfer_bert_weights(keras_mlm, dict(hf_mlm.state_dict()))
        with torch.no_grad():
            hf_logits = hf_mlm(**pt).logits
        k_logits = keras_mlm(k_inputs, training=False)
        d_mlm = max_diff(hf_logits, np.asarray(k_logits))
        print(f"  mlm logits max diff: {d_mlm:.3e}")
        if d_mlm > 1e-3:
            raise ValueError(f"{variant}: BertMaskedLM parity failed")
        path = f"{OUT_DIR}/{variant}_mlm.weights.h5"
        keras_mlm.save_weights(path)
        print(f"  saved -> {path}")

        del hf_model, hf_mlm, keras_model, keras_mlm
        keras.backend.clear_session()
        gc.collect()
