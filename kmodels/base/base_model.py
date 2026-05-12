import json

import keras
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from kmodels.weight_utils import download_file

_HF_PREFIX = "hf:"


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


def load_hf_state_dict(hf_id):
    """Download HF model weights and return a flat ``{name: numpy_array}`` dict.

    Tries (in order):

    1. ``model.safetensors`` (single-file safetensors)
    2. ``model.safetensors.index.json`` (sharded safetensors)
    3. ``pytorch_model.bin`` (single-file pickle)
    4. ``pytorch_model.bin.index.json`` (sharded pickle)
    """
    try:
        path = hf_hub_download(hf_id, "model.safetensors")
    except EntryNotFoundError:
        path = None
    if path is not None:
        from safetensors.numpy import load_file

        return load_file(path)

    try:
        index_path = hf_hub_download(hf_id, "model.safetensors.index.json")
    except EntryNotFoundError:
        index_path = None
    if index_path is not None:
        from safetensors.numpy import load_file

        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = hf_hub_download(hf_id, shard_file)
            state_dict.update(load_file(shard_path))
        return state_dict

    try:
        path = hf_hub_download(hf_id, "pytorch_model.bin")
    except EntryNotFoundError:
        path = None
    if path is not None:
        import torch

        sd = torch.load(path, map_location="cpu", weights_only=True)
        return {k: v.cpu().numpy() if hasattr(v, "cpu") else v for k, v in sd.items()}

    try:
        index_path = hf_hub_download(hf_id, "pytorch_model.bin.index.json")
    except EntryNotFoundError:
        index_path = None
    if index_path is not None:
        import torch

        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = hf_hub_download(hf_id, shard_file)
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
    directly. Subclasses provide a ``config_from_hf`` method that maps
    the parsed ``config.json`` dict into ``__init__`` kwargs, and a
    ``transfer_from_hf`` method that applies the HF state-dict to the
    Keras layers.

    .. code-block:: python

        class OwlViTDetect(BaseModel):
            KMODELS_CONFIG = OWLVIT_CONFIG
            KMODELS_WEIGHTS = OWLVIT_WEIGHTS

            @classmethod
            def config_from_hf(cls, hf_config: dict): ...

            @classmethod
            def transfer_from_hf(cls, model, state_dict): ...

    Usage:

    .. code-block:: python

        m = OwlViTDetect.from_weights("owlvit-base-patch32")

        m = OwlViTDetect.from_weights("hf:google/owlvit-base-patch32")
        m = OwlViTDetect.from_weights("hf:alice/owlvit-finetune")

        m = OwlViTDetect.from_weights("owlvit-base-patch32", load_weights=False)
    """

    KMODELS_CONFIG = None
    KMODELS_WEIGHTS = None
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
            return cls.from_hf(hf_id, load_weights=load_weights, **kwargs)
        return cls.from_release(identifier, load_weights=load_weights, **kwargs)

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
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
        config.update(kwargs)
        model = cls(**config)

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
            if url.lower().endswith(".json"):
                json_path = download_file(url)
                with open(json_path, "r") as f:
                    index = json.load(f)
                if "weight_map" not in index:
                    raise ValueError(
                        f"Sharded weights index '{url}' must contain 'weight_map'."
                    )
                base_url = "/".join(url.split("/")[:-1])
                for shard_file in sorted(set(index["weight_map"].values())):
                    download_file(f"{base_url}/{shard_file}")
                model.load_weights(json_path)
            else:
                weights_path = download_file(url)
                model.load_weights(weights_path)

        return model

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, **kwargs):
        with open(hf_hub_download(hf_id, "config.json"), "r") as f:
            hf_config = json.load(f)
        cls.assert_hf_model_type(hf_id, hf_config)
        kmodels_kwargs = cls.config_from_hf(hf_config)
        kmodels_kwargs.update(kwargs)
        model = cls(**kmodels_kwargs)
        if load_weights:
            state_dict = load_hf_state_dict(hf_id)
            cls.transfer_from_hf(model, state_dict)
        return model

    @classmethod
    def assert_hf_model_type(cls, hf_id, hf_config):
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
    def config_from_hf(cls, hf_config):
        """Map a HuggingFace ``config.json`` dict to ``cls.__init__`` kwargs.

        ``hf_config`` is the result of ``json.load(open("config.json"))``
        — a plain dict, not a ``transformers`` config object. Subclasses
        must override this to support ``"hf:"`` loading.
        """
        raise NotImplementedError(f"{cls.__name__}.config_from_hf is not implemented.")

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        """Transfer weights from an HF ``state_dict`` into ``keras_model``.

        ``hf_state_dict`` is a flat ``{name: numpy_array}`` mapping.
        Subclasses must override this to support ``"hf:"`` loading.
        """
        raise NotImplementedError(
            f"{cls.__name__}.transfer_from_hf is not implemented."
        )
