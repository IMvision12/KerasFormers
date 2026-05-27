from typing import Dict

import numpy as np

from kerasformers.models.roberta.convert_roberta_hf_to_keras import (
    transfer_roberta_weights,
)


def transfer_xlm_roberta_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    transfer_roberta_weights(keras_model, hf_state_dict)


if __name__ == "__main__":
    import gc
    import os

    import keras
    import torch
    from transformers import XLMRobertaForMaskedLM
    from transformers import XLMRobertaModel as HFXLMRobertaModel

    from kerasformers.models.xlm_roberta import XLMRobertaMaskedLM, XLMRobertaModel
    from kerasformers.models.xlm_roberta.config import (
        XLM_ROBERTA_MODEL_CONFIG,
        XLM_ROBERTA_WEIGHT_CONFIG,
    )

    HF_TOKEN = os.environ.get("HF_TOKEN")

    HF_SOURCES = {
        "xlm_roberta_base": "FacebookAI/xlm-roberta-base",
        "xlm_roberta_large": "FacebookAI/xlm-roberta-large",
    }
    SHARD_THRESHOLD_GB = 1.9

    rng = np.random.default_rng(0)

    for variant, meta in XLM_ROBERTA_WEIGHT_CONFIG.items():
        arch = XLM_ROBERTA_MODEL_CONFIG[meta["model"]]
        hf_id = HF_SOURCES[variant]
        pad = arch["pad_token_id"]
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        ids = rng.integers(5, arch["vocab_size"], (2, 16)).astype("int64")
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

        hf_model = HFXLMRobertaModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        keras_model = XLMRobertaModel(**arch)
        transfer_xlm_roberta_weights(keras_model, dict(hf_model.state_dict()))
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
            raise ValueError(f"{variant}: XLMRobertaModel parity failed")
        total_gb = (
            sum(int(np.prod(w.shape)) for w in keras_model.weights) * 4 / (1024**3)
        )
        if total_gb > SHARD_THRESHOLD_GB:
            out_path = f"{variant}.weights.json"
            keras_model.save_weights(out_path, max_shard_size=1.7)
        else:
            out_path = f"{variant}.weights.h5"
            keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path} (~{total_gb:.2f} GiB)")

        hf_mlm = XLMRobertaForMaskedLM.from_pretrained(hf_id, token=HF_TOKEN).eval()
        keras_mlm = XLMRobertaMaskedLM(**arch)
        transfer_xlm_roberta_weights(keras_mlm, dict(hf_mlm.state_dict()))
        with torch.no_grad():
            hf_logits = hf_mlm(**pt).logits
        k_logits = keras_mlm(k_inputs, training=False)
        mlm = k_logits.detach().cpu().numpy()
        d_mlm = float(np.abs(hf_logits.detach().cpu().numpy() - mlm).max())
        print(f"  mlm logits max diff: {d_mlm:.3e}")
        if d_mlm > 1e-3:
            raise ValueError(f"{variant}: XLMRobertaMaskedLM parity failed")
        total_gb = sum(int(np.prod(w.shape)) for w in keras_mlm.weights) * 4 / (1024**3)
        if total_gb > SHARD_THRESHOLD_GB:
            out_path = f"{variant}_mlm.weights.json"
            keras_mlm.save_weights(out_path, max_shard_size=1.7)
        else:
            out_path = f"{variant}_mlm.weights.h5"
            keras_mlm.save_weights(out_path)
        print(f"  Saved -> {out_path} (~{total_gb:.2f} GiB)")

        del hf_model, hf_mlm, keras_model, keras_mlm
        keras.backend.clear_session()
        gc.collect()
