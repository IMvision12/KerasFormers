import hashlib
import json
import os
import re
import warnings

import keras
import numpy as np

from kerasformers.conversion.hf_download_utils import LazyStateDict

SHARD_LIMIT_BYTES = 5 * 1024**3


def cache_root():
    """Root directory for cached converted models.

    ``$KERASFORMERS_HOME/converted`` (else ``~/.cache/kerasformers/converted``),
    self-managed like the HF cache. On an ephemeral box (Colab), point
    ``KERASFORMERS_HOME`` at a persistent mount (Drive) to keep the benefit
    across sessions.
    """
    home = os.environ.get(
        "KERASFORMERS_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "kerasformers"),
    )
    return os.path.join(home, "converted")


def quant_id(quantization):
    """A stable, JSON-safe identity for a quantization spec (or float)."""
    if quantization is None:
        return "float"
    to_dict = getattr(quantization, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return str(quantization)


def cache_dir(cls, identifier, quantization, load_dtype, kwargs):
    """Deterministic per-config cache directory for one converted model.

    Keyed on the model class, source identifier, quantization spec, ``load_dtype``
    and the arch ``**kwargs`` (e.g. ``num_classes`` / ``variant``). Excludes
    ``attn_implementation`` (reapplied at build) and ``low_memory`` (same result).
    """
    payload = {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "identifier": identifier,
        "quantization": quant_id(quantization),
        "load_dtype": load_dtype,
        "kwargs": {k: kwargs[k] for k in sorted(kwargs)},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", identifier)
    return os.path.join(cache_root(), f"{safe_id}.{digest}")


def is_cached(directory):
    """True if ``directory`` holds a complete cache (``meta.json`` is last-written)."""
    return os.path.exists(os.path.join(directory, "meta.json"))


def cache_supported(cls, quantization):
    """Whether ``cls`` can be cache-reloaded from a serialized skeleton.

    Subclassed models rebuild from their constructor config in any precision
    (skeleton → build_for_transfer → stream). Functional models round-trip only
    as float — their built graph can't be re-quantized from a skeleton, so
    functional + quantization is bypassed.
    """
    from kerasformers.base import FunctionalBaseModel, SubclassedBaseModel

    if issubclass(cls, SubclassedBaseModel):
        return True
    if issubclass(cls, FunctionalBaseModel):
        return quantization is None
    return False


def weight_key(model, w):
    """Structural weight path with the model's own top-level name stripped.

    A freshly built model carries the model name in ``w.path``
    (``qwen3_generate/decoder_layers/0/...``) while a deserialized one may not,
    so stripping ``model.name/`` normalizes both to the same stable, unique
    sublayer path — robust to the ``model.weights`` reordering an in-place
    Dense→QuantizedDense swap causes.
    """
    prefix = f"{model.name}/"
    return w.path[len(prefix) :] if w.path.startswith(prefix) else w.path


def save_converted(model, directory, quantization):
    """Cache a converted model's final weights + a rebuild recipe.

    Stores ``model.weights`` as index-keyed sharded safetensors (≤5 GB shards)
    plus ``meta.json`` (serialized config, quant id, per-weight keys + shapes).
    Subclassed models key weights by structural path; functional models by
    position (their auto-numbered layer names differ across a reload). Written
    meta.json last so a partial write is never seen as a cache hit.
    """
    from safetensors.numpy import save_file

    from kerasformers.base import SubclassedBaseModel

    os.makedirs(directory, exist_ok=True)
    weights = list(model.weights)
    arrays = [np.ascontiguousarray(keras.ops.convert_to_numpy(w)) for w in weights]

    keying = "path" if isinstance(model, SubclassedBaseModel) else "index"
    if keying == "path":
        keys = [weight_key(model, w) for w in weights]
        if len(set(keys)) != len(keys):
            keying = "index"
    if keying == "index":
        keys = [f"{i:06d}" for i in range(len(weights))]

    shards = []
    current, current_bytes, shard_idx = {}, 0, 0
    for i, arr in enumerate(arrays):
        if current and current_bytes + arr.nbytes > SHARD_LIMIT_BYTES:
            name = f"weights-{shard_idx:05d}.safetensors"
            save_file(current, os.path.join(directory, name))
            shards.append(name)
            current, current_bytes = {}, 0
            shard_idx += 1
        current[f"{i:06d}"] = arr
        current_bytes += arr.nbytes
    name = f"weights-{shard_idx:05d}.safetensors"
    save_file(current, os.path.join(directory, name))
    shards.append(name)

    config = keras.saving.serialize_keras_object(model)
    if isinstance(model, SubclassedBaseModel):
        # Drop the build recipe so the model deserializes UNBUILT: a quantized
        # reload must rebuild the integer skeleton (quantize_skeleton +
        # build_for_transfer), which only swaps not-yet-built Dense/Embedding.
        # A built-from-config reload would come back float and never match.
        config.pop("build_config", None)

    meta = {
        "config": config,
        "quantization": quant_id(quantization),
        "keying": keying,
        "keys": keys,
        "shapes": [list(w.shape) for w in weights],
        "count": len(weights),
        "shards": shards,
    }
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump(meta, f)


def load_converted(directory, quantization, load_dtype):
    """Rebuild a model from a cache directory and stream its weights back.

    Deserializes the config to a skeleton, re-applies the quantization skeleton
    (subclassed), builds it, then streams each cached tensor onto its weight —
    by structural key (subclassed) or position (functional). Raises on any
    count / shape / key mismatch so the caller can fall back to the source.
    """
    from kerasformers.base.base_mixin import build_dtype_scope

    with open(os.path.join(directory, "meta.json")) as f:
        meta = json.load(f)

    with build_dtype_scope(load_dtype):
        model = keras.saving.deserialize_keras_object(meta["config"])
        if quantization is not None:
            from kerasformers.quantization import quantize_skeleton

            quantize_skeleton(model, quantization)
        if not model.built:
            model.build_for_transfer()

    weights = list(model.weights)
    if len(weights) != meta["count"]:
        raise ValueError(
            f"Cached model has {meta['count']} weights but the rebuilt skeleton "
            f"has {len(weights)}; cache is stale."
        )

    shapes = meta["shapes"]
    if meta["keying"] == "index":
        order = list(range(len(weights)))
    else:
        index_of = {key: i for i, key in enumerate(meta["keys"])}
        order = [index_of[weight_key(model, w)] for w in weights]

    paths = [os.path.join(directory, s) for s in meta["shards"]]
    state = LazyStateDict.from_files(paths)
    try:
        for w, i in zip(weights, order):
            if list(w.shape) != list(shapes[i]):
                raise ValueError(
                    f"Cached weight shape {shapes[i]} != rebuilt {list(w.shape)}; "
                    f"cache is stale."
                )
            w.assign(state[f"{i:06d}"])
    finally:
        state.close()
    return model


def try_load_converted(directory, quantization, load_dtype):
    """Best-effort cache load: return the model, or ``None`` on any failure."""
    try:
        return load_converted(directory, quantization, load_dtype)
    except Exception as e:
        warnings.warn(
            f"Converted-cache reload failed ({e}); rebuilding from source.",
            stacklevel=2,
        )
        return None


def try_save_converted(model, directory, quantization):
    """Best-effort cache save: warn (never raise) if it can't be written."""
    try:
        save_converted(model, directory, quantization)
    except Exception as e:
        warnings.warn(
            f"Could not cache converted model to {directory} ({e}).", stacklevel=2
        )
