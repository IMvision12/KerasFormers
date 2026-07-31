"""Self-describing Hub repos: kf_config.json (model) and kf_preprocessor.json.

Each kerasformers weight repo can carry two small namespaced json files at its
root, next to model.weights.h5:

  kf_config.json        the model: class + flat hyperparameters + weights filename.
  kf_preprocessor.json  the image processor / feature extractor (when present).

They let a model be rebuilt and loaded from the repo alone, so a caller loads by
repo id (kerasformers/detr-resnet-50) instead of a variant hardcoded in the
package. The files are FLAT, transformers style: every hyperparameter sits at the
top level next to ``model_type``, alongside a few kerasformers loader keys
(``model_module`` / ``model_class`` / ``variant`` / ``weights``). The kf_ prefix
avoids the config.json / preprocessor_config.json names that would make the Hub
mis-detect these Keras repos as transformers or keras-hub. The tokenizer stays a
real tokenizer.json and is not described here.
"""

import inspect
import json
import os
from importlib.metadata import PackageNotFoundError, version

CONFIG_FILE = "kf_config.json"
PREPROCESSOR_FILE = "kf_preprocessor.json"

# Top-level keys that are kerasformers loader metadata, not model/preprocessor
# hyperparameters. Everything else in a flat file is a field.
KF_METADATA_KEYS = frozenset(
    {
        "library_name",
        "kerasformers_version",
        "model_module",
        "model_class",
        "preprocessor_module",
        "preprocessor_class",
        "variant",
        "weights",
        "generate_args",
    }
)


def _kf_version():
    try:
        return version("kerasformers")
    except PackageNotFoundError:
        return None


def retuple(config):
    """Restore JSON lists to tuples (used for legacy nested configs)."""

    def cast(value):
        if isinstance(value, list):
            return tuple(cast(item) for item in value)
        return value

    return {key: cast(value) for key, value in config.items()}


def _jsonable(value):
    """Best-effort conversion of a config value to something json can dump."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "tolist"):  # numpy arrays / scalars -> python list / scalar
        return _jsonable(value.tolist())
    return value


def _package_module(cls):
    # class in kerasformers.models.detr.detr_model -> kerasformers.models.detr
    return cls.__module__.rsplit(".", 1)[0]


def model_config_dict(config):
    """Flat config dict (``model_type`` + hyperparameters) for a config instance."""
    return _jsonable(config.to_dict())


def preprocessor_config(preprocessor):
    """Serialize a preprocessor's constructor state.

    kerasformers image processors do not all implement ``get_config``, so read
    the values straight off the instance by walking the constructor signature:
    every named parameter maps to a same-named attribute (``self.size``,
    ``self.resample``, ...). Missing attributes fall back to the parameter
    default. ``self`` and ``*args`` / ``**kwargs`` are skipped.
    """
    config = {}
    for name, param in inspect.signature(
        type(preprocessor).__init__
    ).parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if hasattr(preprocessor, name):
            config[name] = _jsonable(getattr(preprocessor, name))
        elif param.default is not inspect.Parameter.empty:
            config[name] = _jsonable(param.default)
    return config


def _write(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def write_kf_config(
    dest_dir,
    model_cls,
    variant,
    config,
    weights_filename="model.weights.h5",
    generate_args=None,
):
    """Write a flat kf_config.json for ``variant`` into ``dest_dir``.

    ``config`` is a built :class:`BaseConfig` instance (the per-variant config,
    typically ``model_cls.config_class(**overrides)``).

    ``generate_args`` (a dict of default generation settings, e.g. Whisper's
    ``suppress_tokens``) is written under a nested ``generate_args`` key so the
    repo is self-describing for ``model.generate(...)`` too. It defaults to the
    model class's own ``generate_args`` attribute when not passed explicitly.
    """
    payload = {
        "library_name": "kerasformers",
        "kerasformers_version": _kf_version(),
        "model_module": _package_module(model_cls),
        "model_class": model_cls.__name__,
        "variant": variant,
        "weights": weights_filename,
    }
    payload.update(model_config_dict(config))
    if generate_args is None:
        generate_args = getattr(model_cls, "generate_args", None)
    if generate_args:
        payload["generate_args"] = _jsonable(dict(generate_args))
    return _write(os.path.join(dest_dir, CONFIG_FILE), payload)


def write_kf_preprocessor(dest_dir, preprocessor, variant):
    """Write a flat kf_preprocessor.json for a built ``preprocessor`` instance."""
    payload = {
        "library_name": "kerasformers",
        "kerasformers_version": _kf_version(),
        "preprocessor_module": _package_module(type(preprocessor)),
        "preprocessor_class": type(preprocessor).__name__,
        "variant": variant,
    }
    payload.update(preprocessor_config(preprocessor))
    return _write(os.path.join(dest_dir, PREPROCESSOR_FILE), payload)


def _download_repo_json(repo_id, filename):
    """Return the parsed json file from a Hub repo, or None if it is absent."""
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    try:
        path = hf_hub_download(repo_id, filename, token=token)
    except Exception:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_kf_config(repo_id):
    """Fetch and parse kf_config.json from ``repo_id`` (None if the repo lacks it)."""
    return _download_repo_json(repo_id, CONFIG_FILE)


def load_kf_preprocessor(repo_id):
    """Fetch and parse kf_preprocessor.json from ``repo_id`` (None if absent)."""
    return _download_repo_json(repo_id, PREPROCESSOR_FILE)
