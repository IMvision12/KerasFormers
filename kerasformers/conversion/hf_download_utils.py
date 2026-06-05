import json
import os

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


def download_hf_state_dict(hf_id, token=None):
    """Download model weights and return a flat ``{name: numpy_array}`` dict.

    ``token`` is forwarded to ``hf_hub_download`` for gated / private repos.
    Safetensors are read straight to numpy on CPU; torch is used only as a CPU
    fallback for legacy ``.bin`` checkpoints (never touches CUDA).

    Tries (in order):

    1. ``model.safetensors`` (single-file safetensors)
    2. ``model.safetensors.index.json`` (sharded safetensors)
    3. ``pytorch_model.bin`` (single-file pickle)
    4. ``pytorch_model.bin.index.json`` (sharded pickle)
    """
    try:
        path = hf_hub_download(hf_id, "model.safetensors", token=token)
    except EntryNotFoundError:
        path = None
    if path is not None:
        from safetensors.numpy import load_file

        return load_file(path)

    try:
        index_path = hf_hub_download(hf_id, "model.safetensors.index.json", token=token)
    except EntryNotFoundError:
        index_path = None
    if index_path is not None:
        from safetensors.numpy import load_file

        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = hf_hub_download(hf_id, shard_file, token=token)
            state_dict.update(load_file(shard_path))
        return state_dict

    try:
        path = hf_hub_download(hf_id, "pytorch_model.bin", token=token)
    except EntryNotFoundError:
        path = None
    if path is not None:
        import torch

        sd = torch.load(path, map_location="cpu", weights_only=True)
        return {k: v.cpu().numpy() if hasattr(v, "cpu") else v for k, v in sd.items()}

    try:
        index_path = hf_hub_download(hf_id, "pytorch_model.bin.index.json", token=token)
    except EntryNotFoundError:
        index_path = None
    if index_path is not None:
        import torch

        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = hf_hub_download(hf_id, shard_file, token=token)
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            state_dict.update(
                {
                    k: v.cpu().numpy() if hasattr(v, "cpu") else v
                    for k, v in shard.items()
                }
            )
        return state_dict

    raise FileNotFoundError(
        f"No supported weights file found in HF repo '{hf_id}'. "
        f"Expected one of: model.safetensors, model.safetensors.index.json, "
        f"pytorch_model.bin, pytorch_model.bin.index.json."
    )


def load_and_convert_from_hf(
    model,
    model_name,
    hf_model_id,
    transfer_fn,
    is_gated=False,
):
    """Download, convert, and cache source weights for a Keras model.

    Generic helper used for models whose Keras weights cannot be
    redistributed directly — either due to **license gating** (SAM3,
    DINOv3) or because the converted weights **exceed distribution host
    limits** (MetaCLIP 2 Huge/Giant variants at 3-4 GB single tensors,
    larger than GitHub's 2 GB release asset cap).

    Weights are cached at ``~/.cache/kerasformers/<model_name>/``. Sharded at
    5 GB per shard for local cache.

    Args:
        model: The Keras model instance to load weights into.
        model_name: String used as the cache subdirectory name.
        hf_model_id: Model-hub identifier.
        transfer_fn: Callable ``(keras_model, hf_state_dict) -> None``. Receives
            the raw on-disk checkpoint tensors (the same key layout the
            ``hf:`` / safetensors release path produces).
        is_gated: When True, emits the license-acceptance error message
            on 401/403. When False (default), lets the download error propagate.
    """
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "kerasformers", model_name
    )
    cached_weights = os.path.join(cache_dir, f"{model_name}.weights.h5")

    if os.path.exists(cached_weights):
        print(f"Loading cached {model_name} weights from {cached_weights}")
        model.load_weights(cached_weights)
        return

    gated_note = " (requires accepted license + HF token)" if is_gated else ""
    print(f"Downloading {model_name} from HuggingFace{gated_note}...")

    hf_token = os.environ.get("HF_TOKEN")
    try:
        hf_state_dict = download_hf_state_dict(hf_model_id, token=hf_token)
    except Exception as e:
        error_msg = str(e)
        if is_gated and (
            "gated" in error_msg.lower() or "401" in error_msg or "403" in error_msg
        ):
            raise OSError(
                f"\n{'=' * 60}\n"
                f"Access denied for '{hf_model_id}'.\n\n"
                f"This model is gated and requires license acceptance.\n"
                f"Please follow these steps:\n\n"
                f"  1. Go to https://huggingface.co/{hf_model_id}\n"
                f"     and accept the license agreement.\n\n"
                f"  2. Authenticate using one of:\n"
                f"     - Run: huggingface-cli login\n"
                f"     - Or set: export HF_TOKEN=<your_token>\n"
                f"{'=' * 60}"
            ) from e
        raise

    print(f"Converting {model_name} weights to Keras...")
    transfer_fn(model, hf_state_dict)

    os.makedirs(cache_dir, exist_ok=True)

    total_bytes = sum(w.numpy().nbytes for w in model.weights)
    size_gb = total_bytes / (1024**3)
    save_kwargs = {}
    if size_gb > 5:
        save_kwargs["max_shard_size"] = 5.0

    model.save_weights(cached_weights, **save_kwargs)
    print(f"Cached {model_name} weights to {cached_weights} ({size_gb:.1f} GB)")
