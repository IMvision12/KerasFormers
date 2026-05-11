import json

import keras

from kmodels.weight_utils import download_file

_HF_PREFIX = "hf:"


def _hf_download(hf_id, filename):
    """Download a single file from a HuggingFace Hub repo.

    Uses ``huggingface_hub.hf_hub_download`` directly so that
    ``transformers`` is **not** required for HF loading.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Loading from HuggingFace requires the `huggingface_hub` package. "
            "Install it with `pip install huggingface_hub`."
        ) from e
    from huggingface_hub.utils import EntryNotFoundError  # noqa: F401

    return hf_hub_download(hf_id, filename)


def _hf_try_download(hf_id, filename):
    """Best-effort download. Returns ``None`` if ``filename`` is absent."""
    try:
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        EntryNotFoundError = Exception  # type: ignore[misc]
    try:
        return _hf_download(hf_id, filename)
    except EntryNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001
        # huggingface_hub raises different exception classes across versions;
        # fall through silently and let the caller try the next candidate.
        if "404" in str(e) or "not found" in str(e).lower():
            return None
        raise


def _load_hf_config(hf_id):
    """Download ``config.json`` from an HF repo and return it as a dict."""
    path = _hf_download(hf_id, "config.json")
    with open(path, "r") as f:
        return json.load(f)


def hf_num_labels(hf_config):
    """Derive ``num_labels`` from a HuggingFace ``config.json`` dict.

    HF's ``PretrainedConfig`` exposes ``num_labels`` as a property derived
    from ``id2label``, but ``config.json`` typically stores ``id2label``
    rather than ``num_labels`` directly. This helper checks both.
    """
    if "num_labels" in hf_config:
        return hf_config["num_labels"]
    id2label = hf_config.get("id2label")
    if id2label:
        return len(id2label)
    label2id = hf_config.get("label2id")
    if label2id:
        return len(label2id)
    raise KeyError(
        "Could not determine num_labels from HF config.json — "
        "neither 'num_labels' nor 'id2label' / 'label2id' is present."
    )


def _load_hf_state_dict(hf_id):
    """Download HF model weights and return a flat ``{name: numpy_array}`` dict.

    Tries (in order):

    1. ``model.safetensors`` (single-file safetensors)
    2. ``model.safetensors.index.json`` (sharded safetensors)
    3. ``pytorch_model.bin`` (single-file pickle)
    4. ``pytorch_model.bin.index.json`` (sharded pickle)
    """
    # 1. Single-file safetensors
    path = _hf_try_download(hf_id, "model.safetensors")
    if path is not None:
        from safetensors.numpy import load_file

        return load_file(path)

    # 2. Sharded safetensors
    index_path = _hf_try_download(hf_id, "model.safetensors.index.json")
    if index_path is not None:
        from safetensors.numpy import load_file

        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = _hf_download(hf_id, shard_file)
            state_dict.update(load_file(shard_path))
        return state_dict

    # 3. Single-file pytorch_model.bin
    path = _hf_try_download(hf_id, "pytorch_model.bin")
    if path is not None:
        import torch

        sd = torch.load(path, map_location="cpu", weights_only=True)
        return {k: v.cpu().numpy() if hasattr(v, "cpu") else v for k, v in sd.items()}

    # 4. Sharded pytorch_model.bin
    index_path = _hf_try_download(hf_id, "pytorch_model.bin.index.json")
    if index_path is not None:
        import torch

        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = _hf_download(hf_id, shard_file)
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


class BaseModel(keras.Model):
    """Base class for kmodels models with unified weight loading.

    Subclasses are Functional Keras models that share a single entry
    point for loading pretrained weights, regardless of source:

    1. **kmodels release** — weights hosted on GitHub Releases keyed by
       a short variant string (e.g. ``"owlvit-base-patch32"``).
    2. **HuggingFace** — weights pulled from a HF Hub repo, identified
       by an ``"hf:org/repo"`` string. Works for original HF checkpoints
       and for community fine-tunes that share the same architecture.

    HF loading uses ``huggingface_hub`` (not ``transformers``) — it
    downloads ``config.json`` and the safetensors / pytorch weights
    directly. Subclasses provide a ``_config_from_hf`` method that maps
    the parsed ``config.json`` dict into ``__init__`` kwargs, and a
    ``_transfer_from_hf`` method that applies the HF state-dict to the
    Keras layers.

    .. code-block:: python

        class OwlViTDetect(BaseModel):
            KMODELS_CONFIG = OWLVIT_CONFIG       # variant -> kwargs
            KMODELS_WEIGHTS = OWLVIT_WEIGHTS     # variant -> {"url": ...}

            @classmethod
            def _config_from_hf(cls, hf_config: dict): ...

            @classmethod
            def _transfer_from_hf(cls, model, state_dict): ...

    Usage:

    .. code-block:: python

        # Trained kmodels release
        m = OwlViTDetect.from_weights("owlvit-base-patch32")

        # Trained HF original or fine-tune
        m = OwlViTDetect.from_weights("hf:google/owlvit-base-patch32")
        m = OwlViTDetect.from_weights("hf:alice/owlvit-finetune")

        # Untrained
        m = OwlViTDetect.from_weights("owlvit-base-patch32", load_weights=False)
    """

    KMODELS_CONFIG = None
    KMODELS_WEIGHTS = None
    # ``model_type`` value(s) (from HF ``config.json``) this class can load
    # from HuggingFace. ``str`` for a single value or ``tuple[str, ...]`` for
    # several aliases (e.g. ``("d_fine", "dfine")``). ``None`` disables the
    # guard — only do that for classes that don't load from HF at all.
    HF_MODEL_TYPE = None

    @classmethod
    def from_weights(cls, identifier, load_weights=True, **kwargs):
        """Build a model and (optionally) load pretrained weights.

        Args:
            identifier: Either a kmodels variant string (e.g.
                ``"owlvit-base-patch32"``) which resolves against
                ``cls.KMODELS_CONFIG`` / ``cls.KMODELS_WEIGHTS``, or an
                ``"hf:<org>/<repo>"`` string which pulls config and
                weights from HuggingFace.
            load_weights: If ``False``, only the architecture is built
                (random init). For HF ids, ``config.json`` is still
                fetched to size the model; the weight files are not.
            **kwargs: Forwarded to the model constructor.

        Returns:
            An initialized model instance.
        """
        if identifier.startswith(_HF_PREFIX):
            hf_id = identifier[len(_HF_PREFIX) :]
            return cls._from_hf(hf_id, load_weights=load_weights, **kwargs)
        return cls._from_release(identifier, load_weights=load_weights, **kwargs)

    @classmethod
    def _from_release(cls, variant, load_weights=True, **kwargs):
        if cls.KMODELS_CONFIG is None:
            raise NotImplementedError(
                f"{cls.__name__} must set KMODELS_CONFIG to use from_weights()."
            )
        if variant not in cls.KMODELS_CONFIG:
            available = sorted(cls.KMODELS_CONFIG.keys())
            raise ValueError(
                f"Unknown variant '{variant}' for {cls.__name__}. "
                f"Available variants: {available}"
            )

        config = dict(cls.KMODELS_CONFIG[variant])
        model = cls(**config, **kwargs)

        if load_weights:
            if cls.KMODELS_WEIGHTS is None or variant not in cls.KMODELS_WEIGHTS:
                raise ValueError(
                    f"No release weights configured for variant '{variant}'. "
                    f"Pass load_weights=False to build an untrained model."
                )
            url = cls.KMODELS_WEIGHTS[variant]
            if isinstance(url, dict):
                url = url.get("url")
            if not url:
                raise ValueError(
                    f"Release weights URL for variant '{variant}' is empty."
                )
            weights_path = download_file(url)
            model.load_weights(weights_path)

        return model

    @classmethod
    def _from_hf(cls, hf_id, load_weights=True, **kwargs):
        hf_config = _load_hf_config(hf_id)
        cls._assert_hf_model_type(hf_id, hf_config)
        kmodels_kwargs = cls._config_from_hf(hf_config)
        kmodels_kwargs.update(kwargs)
        model = cls(**kmodels_kwargs)
        if load_weights:
            state_dict = _load_hf_state_dict(hf_id)
            cls._transfer_from_hf(model, state_dict)
        return model

    @classmethod
    def _assert_hf_model_type(cls, hf_id, hf_config):
        """Reject HF configs whose ``model_type`` doesn't match this class.

        Fails fast with a clear message instead of letting the user wait
        for a ``KeyError`` or shape mismatch deep inside weight transfer.
        Subclasses opt in by setting ``cls.HF_MODEL_TYPE``; the check is
        skipped when it's ``None``.
        """
        expected = cls.HF_MODEL_TYPE
        if expected is None:
            return
        if isinstance(expected, str):
            expected = (expected,)
        actual = hf_config.get("model_type")
        if actual not in expected:
            options = expected[0] if len(expected) == 1 else f"one of {list(expected)}"
            raise ValueError(
                f"{cls.__name__} can only load HF models whose "
                f"config.json model_type is {options}, but '{hf_id}' "
                f"has model_type={actual!r}. This kmodels class is the "
                f"wrong destination for that checkpoint."
            )

    @classmethod
    def _config_from_hf(cls, hf_config):
        """Map a HuggingFace ``config.json`` dict to ``cls.__init__`` kwargs.

        ``hf_config`` is the result of ``json.load(open("config.json"))``
        — a plain dict, not a ``transformers`` config object. Subclasses
        must override this to support ``"hf:"`` loading.
        """
        raise NotImplementedError(f"{cls.__name__}._config_from_hf is not implemented.")

    @classmethod
    def _transfer_from_hf(cls, keras_model, hf_state_dict):
        """Transfer weights from an HF ``state_dict`` into ``keras_model``.

        ``hf_state_dict`` is a flat ``{name: numpy_array}`` mapping.
        Subclasses must override this to support ``"hf:"`` loading.
        """
        raise NotImplementedError(
            f"{cls.__name__}._transfer_from_hf is not implemented."
        )
