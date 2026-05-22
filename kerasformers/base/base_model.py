import json

import keras
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from kerasformers.weight_utils import download_file

_HF_PREFIX = "hf:"


def hf_num_labels(hf_config):
    """Derive ``num_labels`` from a ``config.json`` dict.

    A serialized ``config.json`` typically stores ``id2label`` rather than
    ``num_labels`` directly, so this helper derives the label count from
    whichever of ``num_labels`` / ``id2label`` / ``label2id`` is present.
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


def download_hf_state_dict(hf_id):
    """Download model weights and return a flat ``{name: numpy_array}`` dict.

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
    """Base class for kerasformers models with unified weight loading.

    Subclasses are Functional Keras models that share a single entry
    point for loading pretrained weights, regardless of source:

    1. **kerasformers release** — weights hosted on GitHub Releases keyed by
       a short variant string (e.g. ``"owlvit-base-patch32"``).
    2. **Hub** — weights pulled from a model-hub repo, identified
       by an ``"hf:org/repo"`` string. Works for the original checkpoints
       and for community fine-tunes that share the same architecture.

    Hub loading uses ``huggingface_hub`` (not ``transformers``) — it
    downloads ``config.json`` and the safetensors / pytorch weights
    directly. Subclasses provide a ``config_from_hf`` method that maps
    the parsed ``config.json`` dict into ``__init__`` kwargs, and a
    ``transfer_from_hf`` method that applies the source state-dict to the
    Keras layers.

    .. code-block:: python

        class OwlViTDetect(BaseModel):
            BASE_MODEL_CONFIG = OWLVIT_CONFIG
            BASE_WEIGHT_CONFIG = OWLVIT_WEIGHTS

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

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = None

    @classmethod
    def from_weights(cls, identifier, load_weights=True, skip_mismatch=False, **kwargs):
        """Build a model and (optionally) load pretrained weights.

        Args:
            identifier: One of two forms:

                * a kerasformers variant string (e.g. ``"resnet50_a1_in1k"``)
                  — resolves against ``cls.BASE_MODEL_CONFIG`` /
                  ``cls.BASE_WEIGHT_CONFIG``.
                * ``"hf:<org>/<repo>"`` — pulls config and weights from
                  the model Hub. Dispatches to :meth:`from_hf`, which
                  handles both transformers-style repos (CLIP, SigLIP,
                  DETR, …) and timm-style repos
                  (``hf:timm/resnet50.a1_in1k``).

            load_weights: If ``False``, only the architecture is built
                (random init). For ``hf:`` ids, ``config.json`` is still
                fetched to size the model; the weight files are not.
            skip_mismatch: If ``True``, layers whose shape in the
                checkpoint disagrees with the instantiated model are
                skipped during weight load and left at their default
                initialization. Useful for fine-tuning: pass
                ``num_classes=N, skip_mismatch=True`` to swap in a new
                classifier head while loading the rest of the backbone.
                Only applied on the kerasformers-release path (``.h5`` /
                ``.json`` URLs); ``hf:`` paths go through hand-mapped
                transfer functions and ignore this flag.
            **kwargs: Forwarded to the model constructor (or to
                ``from_hf`` when applicable).

        Returns:
            An initialized model instance.
        """
        if identifier.startswith(_HF_PREFIX):
            hf_id = identifier[len(_HF_PREFIX) :]
            return cls.from_hf(hf_id, load_weights=load_weights, **kwargs)
        return cls.from_release(
            identifier,
            load_weights=load_weights,
            skip_mismatch=skip_mismatch,
            **kwargs,
        )

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        """Map a timm state-dict onto ``keras_model``'s weights.

        Default raises :class:`NotImplementedError`. Subclasses opt in
        by implementing the per-family timm-name → keras-weight mapping
        (typically delegating to a module-level
        ``transfer_<family>_weights`` function). Reached via
        :meth:`from_hf` when ``HF_MODEL_TYPE is None`` (i.e., the
        family loads from timm-style repos, not transformers-style
        ones).
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support loading from timm-style HF "
            f"repos. Implement `transfer_from_timm` to enable it."
        )

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        if cls.BASE_MODEL_CONFIG is None:
            raise NotImplementedError(
                f"{cls.__name__} must set BASE_MODEL_CONFIG to use from_weights()."
            )
        if variant not in cls.BASE_MODEL_CONFIG:
            available = sorted(cls.BASE_MODEL_CONFIG.keys())
            raise ValueError(
                f"Unknown variant '{variant}' for {cls.__name__}. "
                f"Available variants: {available}"
            )

        config = dict(cls.BASE_MODEL_CONFIG[variant])
        config.update(kwargs)
        model = cls(**config)

        if load_weights:
            if cls.BASE_WEIGHT_CONFIG is None or variant not in cls.BASE_WEIGHT_CONFIG:
                raise ValueError(
                    f"No release weights configured for variant '{variant}'. "
                    f"Pass load_weights=False to build an untrained model."
                )
            entry = cls.BASE_WEIGHT_CONFIG[variant]
            if isinstance(entry, dict):
                hf_id = entry.get("hf_id")
                gated = entry.get("gated", False)
                url = entry.get("url")
                hf_model_cls = entry.get("hf_model_cls")
                hf_kwargs = entry.get("hf_kwargs")
            else:
                hf_id = None
                gated = False
                url = entry
                hf_model_cls = None
                hf_kwargs = None

            if hf_id:
                from kerasformers.weight_utils.hf_gated_weight_download import (
                    load_and_convert_from_hf,
                )

                load_and_convert_from_hf(
                    model=model,
                    model_name=variant,
                    hf_model_id=hf_id,
                    transfer_fn=cls.transfer_from_hf,
                    is_gated=gated,
                    hf_model_cls=hf_model_cls,
                    hf_kwargs=hf_kwargs,
                )
            elif url:
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
                    model.load_weights(json_path, skip_mismatch=skip_mismatch)
                else:
                    weights_path = download_file(url)
                    model.load_weights(weights_path, skip_mismatch=skip_mismatch)
            else:
                raise ValueError(
                    f"Release weights entry for variant '{variant}' has "
                    f"neither 'url' nor 'hf_id'."
                )

        return model

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, variant=None, **kwargs):
        """Load a model from a model-hub repo.

        Two flavours, auto-detected by :attr:`HF_MODEL_TYPE`:

        1. **Transformers-style repos** (``HF_MODEL_TYPE`` set — CLIP,
           SigLIP, DETR, EoMT, …): pulls ``config.json``, validates
           ``model_type``, builds via :meth:`config_from_hf`, and
           dispatches to :meth:`transfer_from_hf`.
        2. **Timm-style repos** (``HF_MODEL_TYPE is None`` — ResNet,
           ConvNeXt, EfficientNet, …): infers the kerasformers
           variant from the repo's trailing path segment, builds via
           :attr:`BASE_MODEL_CONFIG`, and dispatches to
           :meth:`transfer_from_timm`. No ``config.json`` is parsed
           (timm checkpoints don't carry a transformers-style
           ``model_type``).

        Args:
            hf_id: Model-hub id, e.g.
                ``"openai/clip-vit-base-patch16"`` (transformers-style)
                or ``"timm/resnet50.a1_in1k"`` (timm-style).
            load_weights: If ``False``, only the architecture is built.
            variant: For timm-style repos, override the inferred
                kerasformers variant id (e.g., for community fine-tunes
                whose repo name doesn't follow the timm convention).
                Ignored for transformers-style repos.
            **kwargs: Forwarded to the model constructor.

        Returns:
            An initialized model instance.
        """
        if cls.HF_MODEL_TYPE is None:
            if variant is None:
                tail = hf_id.split("/")[-1]
                stem = tail.replace(".", "_")
                for candidate in cls.BASE_MODEL_CONFIG or {}:
                    if stem == candidate or stem.startswith(candidate + "_"):
                        variant = candidate
                        break
                if variant is None:
                    raise ValueError(
                        f"Cannot infer kerasformers variant from hf_id "
                        f"'{hf_id}'. Pass `variant=` explicitly. Available "
                        f"variants: {sorted(cls.BASE_MODEL_CONFIG or {})}"
                    )
            model = cls.from_release(variant, load_weights=False, **kwargs)
            if load_weights:
                state_dict = download_hf_state_dict(hf_id)
                cls.transfer_from_timm(model, state_dict)
            return model

        with open(hf_hub_download(hf_id, "config.json"), "r") as f:
            hf_config = json.load(f)
        cls.assert_hf_model_type(hf_id, hf_config)
        kerasformers_kwargs = cls.config_from_hf(hf_config)
        kerasformers_kwargs.update(kwargs)
        model = cls(**kerasformers_kwargs)
        if load_weights:
            state_dict = download_hf_state_dict(hf_id)
            cls.transfer_from_hf(model, state_dict)
        return model

    @classmethod
    def assert_hf_model_type(cls, hf_id, hf_config):
        """Reject configs whose ``model_type`` doesn't match this class.

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
                f"has model_type={actual!r}. This kerasformers class is the "
                f"wrong destination for that checkpoint."
            )

    @classmethod
    def config_from_hf(cls, hf_config):
        """Map a ``config.json`` dict to ``cls.__init__`` kwargs.

        ``hf_config`` is the result of ``json.load(open("config.json"))``
        — a plain dict, not a ``transformers`` config object. Subclasses
        must override this to support ``"hf:"`` loading.
        """
        raise NotImplementedError(f"{cls.__name__}.config_from_hf is not implemented.")

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        """Transfer weights from a source ``state_dict`` into ``keras_model``.

        ``hf_state_dict`` is a flat ``{name: numpy_array}`` mapping.
        Subclasses must override this to support ``"hf:"`` loading.
        """
        raise NotImplementedError(
            f"{cls.__name__}.transfer_from_hf is not implemented."
        )
