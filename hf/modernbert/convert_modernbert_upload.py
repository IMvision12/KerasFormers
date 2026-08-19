"""Convert the upstream ModernBERT checkpoints to kerasformers and (re)host them
on the Hugging Face Hub as kerasformers/modernbert_{base,large}.

Each repo gets the current single-file layout:
  - model.weights.h5   (the FULL masked-LM superset, saved from ModernBertMaskedLM;
                        every ModernBert* class loads its own subset via CHECKPOINT_SOURCE)
  - kf_config.json     (declares the canonical ModernBertModel + config)
  - tokenizer.json     (the upstream fast-tokenizer spec)

This is user-run: it does Hub uploads. Set HF_TOKEN in the environment (never hardcode
it), keep DRY_RUN=True for a first pass to confirm conversion + staging, then flip it to
False to actually create/replace the repos.

Run with the torch backend so the weight transfer works:
    KERAS_BACKEND=torch HF_TOKEN=hf_... python hf/modernbert/convert_modernbert_upload.py
"""

import os
import shutil
import tempfile

import numpy as np

# ----------------------------------------------------------------------------- config
DRY_RUN = (
    True  # True = convert + stage locally, no Hub writes. Flip to False to upload.
)
RUN_PARITY = True  # cosine-gate each conversion against the HF reference before saving.
HF_TOKEN = os.environ.get(
    "HF_TOKEN"
)  # never hardcode; export HF_TOKEN=... in the shell
ORG = "kerasformers"

VARIANTS = {
    "modernbert_base": "answerdotai/ModernBERT-base",
    "modernbert_large": "answerdotai/ModernBERT-large",
}

COSINE_GATE = 0.9999  # deep fp32 accumulation: gate on cosine, not max-abs (see docs)


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_readme(variant, hf_id, arch):
    lines = [
        "---",
        "library_name: kerasformers",
        "license: apache-2.0",
        "pipeline_tag: fill-mask",
        "tags:",
        "  - keras",
        "  - modernbert",
        f"base_model: {hf_id}",
        "---",
        "",
        f"# {variant}",
        "",
        "ModernBERT converted to [kerasformers](https://github.com/IMvision12/KerasFormers)",
        "(pure Keras 3, runnable on JAX / PyTorch / TensorFlow). Converted from"
        f" [`{hf_id}`](https://huggingface.co/{hf_id}).",
        "",
        "```python",
        "from kerasformers.models.modernbert import ModernBertModel, ModernBertTokenizer",
        "",
        f'model = ModernBertModel.from_weights("{ORG}/{variant}")',
        f'tokenizer = ModernBertTokenizer.from_weights("{ORG}/{variant}")',
        'out = model(tokenizer("Hello, world."))["last_hidden_state"]',
        "```",
        "",
        "| layers | embed_dim | heads | mlp_dim |",
        "|---|---|---|---|",
        f"| {arch['num_layers']} | {arch['embed_dim']} |"
        f" {arch['num_heads']} | {arch['mlp_dim']} |",
        "",
        "The single `model.weights.h5` is the full masked-LM checkpoint; the encoder,",
        "masked-LM, and task-head classes each load their own subset.",
        "",
    ]
    return "\n".join(lines)


def convert(variant, hf_id, dest_dir):
    """Convert hf_id -> ModernBertMaskedLM superset, save model.weights.h5 + kf_config
    + tokenizer.json into dest_dir. Returns the backbone arch dict."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import ModernBertForMaskedLM

    from kerasformers.conversion.kf_config import write_kf_config
    from kerasformers.models.modernbert import (
        ModernBertConfig,
        ModernBertMaskedLM,
        ModernBertModel,
    )
    from kerasformers.models.modernbert.convert_modernbert_hf_to_keras import (
        transfer_modernbert_weights,
    )

    print(f"  loading {hf_id} ...")
    hf_mlm = ModernBertForMaskedLM.from_pretrained(hf_id, token=HF_TOKEN).eval()
    arch = ModernBertModel.config_from_hf(hf_mlm.config.to_dict())

    keras_mlm = ModernBertMaskedLM(**arch)
    transfer_modernbert_weights(keras_mlm, dict(hf_mlm.state_dict()))

    if RUN_PARITY:
        rng = np.random.default_rng(0)
        ids = rng.integers(0, arch["vocab_size"], (2, 160)).astype("int64")
        mask = np.ones((2, 160), dtype="int64")
        mask[0, 150:] = 0
        pt = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
        }
        with torch.no_grad():
            hf_logits = hf_mlm(**pt).logits.detach().cpu().numpy()
        k = keras_mlm(
            {"input_ids": ids.astype("int32"), "attention_mask": mask.astype("int32")},
            training=False,
        )
        k = k.detach().cpu().numpy() if hasattr(k, "detach") else np.asarray(k)
        c = cosine(hf_logits, k)
        print(f"  mlm parity: max|d|={np.abs(hf_logits - k).max():.3e}  cosine={c:.7f}")
        if c < COSINE_GATE:
            raise ValueError(f"{variant}: parity failed (cosine {c:.5f})")

    weights_path = os.path.join(dest_dir, "model.weights.h5")
    keras_mlm.save_weights(weights_path)
    print(f"  saved {weights_path}")

    # kf_config declares the canonical ModernBertModel (backbone superset owner);
    # ModernBERT weights are float32.
    config = ModernBertConfig(**arch)
    write_kf_config(dest_dir, ModernBertModel, variant, config, weight_dtype="float32")

    tok_src = hf_hub_download(hf_id, "tokenizer.json", token=HF_TOKEN)
    shutil.copy(tok_src, os.path.join(dest_dir, "tokenizer.json"))

    with open(os.path.join(dest_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(build_readme(variant, hf_id, arch))

    return arch


def upload(variant, dest_dir):
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    repo_id = f"{ORG}/{variant}"
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=dest_dir,
        commit_message=f"Add kerasformers {variant} (ModernBERT)",
    )
    print(f"  uploaded -> https://huggingface.co/{repo_id}")


def main():
    if not HF_TOKEN:
        raise SystemExit("Set HF_TOKEN in the environment (do not hardcode it).")

    if not DRY_RUN:
        from huggingface_hub import HfApi

        who = HfApi(token=HF_TOKEN).whoami()
        print(f"authenticated as: {who.get('name')}")

    for variant, hf_id in VARIANTS.items():
        print(f"\n{'=' * 64}\n{variant}  <-  {hf_id}\n{'=' * 64}")
        dest = tempfile.mkdtemp(prefix=f"{variant}_")
        try:
            convert(variant, hf_id, dest)
            if DRY_RUN:
                print(f"  [DRY_RUN] staged in {dest} (set DRY_RUN=False to upload)")
                print(f"  files: {sorted(os.listdir(dest))}")
            else:
                upload(variant, dest)
        finally:
            if not DRY_RUN:
                shutil.rmtree(dest, ignore_errors=True)


if __name__ == "__main__":
    main()
