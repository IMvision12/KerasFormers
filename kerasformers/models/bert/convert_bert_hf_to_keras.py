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
    "classifier/kernel": "classifier.weight",
    "classifier/bias": "classifier.bias",
    "qa_outputs/kernel": "qa_outputs.weight",
    "qa_outputs/bias": "qa_outputs.bias",
    "nsp_classifier/kernel": "cls.seq_relationship.weight",
    "nsp_classifier/bias": "cls.seq_relationship.bias",
}

_OPTIONAL_HEADS = ("classifier", "qa_outputs", "nsp_classifier")


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
    if key.startswith("bert."):
        key = key[len("bert.") :]
    return key.replace("LayerNorm.gamma", "LayerNorm.weight").replace(
        "LayerNorm.beta", "LayerNorm.bias"
    )


def transfer_bert_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
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
    from transformers import BertForMaskedLM
    from transformers import BertModel as HFBertModel

    from kerasformers.models.bert import BertMaskedLM, BertModel
    from kerasformers.models.bert.config import BERT_MODEL_CONFIG, BERT_WEIGHT_CONFIG

    HF_TOKEN = os.environ.get("HF_TOKEN")

    # Detach both sides before comparing parity: Keras torch-backend outputs
    # require grad, and np.asarray would call .numpy() and fail.
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

        hf_model = HFBertModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        keras_model = BertModel(**arch)
        transfer_bert_weights(keras_model, dict(hf_model.state_dict()))
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
        # Looser than base (~1e-5): float32 accumulation over the deeper/wider
        # large stack (24 layers) lands ~1.5e-4 vs HF's eager reference.
        if max(d_seq, d_pool) > 1e-3:
            raise ValueError(f"{variant}: BertModel parity failed")
        total_params = sum(int(np.prod(w.shape)) for w in keras_model.weights)
        total_gb = (total_params * 4) / (1024**3)
        if total_gb > 1.7:
            out_path = f"{variant}.weights.json"
            keras_model.save_weights(out_path, max_shard_size=1.7)
            print(f"  Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
        else:
            out_path = f"{variant}.weights.h5"
            keras_model.save_weights(out_path)
            print(f"  Saved -> {out_path} (~{total_gb:.2f} GB)")

        hf_mlm = BertForMaskedLM.from_pretrained(hf_id, token=HF_TOKEN).eval()
        keras_mlm = BertMaskedLM(**arch)
        transfer_bert_weights(keras_mlm, dict(hf_mlm.state_dict()))
        with torch.no_grad():
            hf_logits = hf_mlm(**pt).logits
        k_logits = keras_mlm(k_inputs, training=False)
        mlm = k_logits.detach().cpu().numpy()
        d_mlm = float(np.abs(hf_logits.detach().cpu().numpy() - mlm).max())
        print(f"  mlm logits max diff: {d_mlm:.3e}")
        if d_mlm > 1e-3:
            raise ValueError(f"{variant}: BertMaskedLM parity failed")
        total_params = sum(int(np.prod(w.shape)) for w in keras_mlm.weights)
        total_gb = (total_params * 4) / (1024**3)
        if total_gb > 1.7:
            out_path = f"{variant}_mlm.weights.json"
            keras_mlm.save_weights(out_path, max_shard_size=1.7)
            print(f"  Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
        else:
            out_path = f"{variant}_mlm.weights.h5"
            keras_mlm.save_weights(out_path)
            print(f"  Saved -> {out_path} (~{total_gb:.2f} GB)")

        del hf_model, hf_mlm, keras_model, keras_mlm
        keras.backend.clear_session()
        gc.collect()
