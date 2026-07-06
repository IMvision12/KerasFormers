import contextlib
import inspect
import json

import keras
from huggingface_hub import hf_hub_download

from kerasformers.base import base_attention
from kerasformers.conversion import download_file
from kerasformers.conversion.hf_download_utils import (
    download_hf_state_dict,
)
from kerasformers.conversion.weight_transfer_util import (
    skip_mismatched_weights,
)

_HF_PREFIX = "hf:"


def warn_skipped(skipped):
    """Print a note about weights left at init due to ``skip_mismatch``."""
    if skipped:
        print(
            f"[from_weights] skip_mismatch: left {len(skipped)} weight(s) at their "
            f"initialized values due to shape mismatch (e.g. a resized head): "
            f"{skipped}"
        )


@contextlib.contextmanager
def build_dtype_scope(dtype):
    """Build the model under a global dtype policy (e.g. ``"bfloat16"``).

    A layer's own dtype policy does **not** propagate to sublayers it creates in
    ``__init__`` — a nested ``Dense`` / ``Embedding`` reads the *global* policy at
    construction time, so passing ``dtype=`` to a model constructor still leaves
    its weights float32. Setting the global policy here makes the whole model —
    including the lazy build the converter triggers — allocate its variables in
    ``dtype``; a streamed fp32 / bf16 checkpoint is then cast to it on assign, so
    a bf16 model lands at its native ~2 bytes/param instead of an fp32 upcast.
    ``None`` restores the default (fp32) behaviour as a no-op.
    """
    if dtype is None:
        yield
        return
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy(dtype)
    try:
        yield
    finally:
        keras.config.set_dtype_policy(previous)


class WeightLoadingMixin:
    """Unified pretrained-weight loading API shared by all kerasformers models.

    Mixed into :class:`FunctionalBaseModel` (functional models) and
    :class:`SubclassedBaseModel` (imperative / subclassed models). Kept as a
    plain mixin — **not** a ``keras.Model`` subclass — so those two bases stay
    independent ``keras.Model`` subclasses (see :class:`SubclassedBaseModel`
    for why that independence matters). Subclasses share a single entry
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

        class OwlViTDetect(FunctionalBaseModel):
            BASE_MODEL_CONFIG = OWLVIT_CONFIG
            BASE_WEIGHT_CONFIG = OWLVIT_WEIGHTS_URLS

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
    def from_weights(
        cls,
        identifier,
        load_weights=True,
        skip_mismatch=False,
        attn_implementation=None,
        quantization=None,
        low_memory=False,
        load_dtype=None,
        **kwargs,
    ):
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
            skip_mismatch: If ``True``, weights whose checkpoint shape
                disagrees with the instantiated model are skipped during
                load and left at their default initialization. Useful for
                fine-tuning: pass ``num_classes=N, skip_mismatch=True`` to
                swap in a new classifier head while loading the rest of the
                backbone. Applied on both the kerasformers-release path
                (``.h5`` / ``.json`` ``load_weights``) and the ``hf:`` /
                converter transfer path (mismatched targets left at init).
            attn_implementation: ``"sdpa"`` (portable manual math, the default)
                or ``"flash"`` (``keras.ops.dot_product_attention`` with the
                flash kernel; needs a flash-capable GPU/TPU and fp16/bf16). Set
                before the model is built, so it applies to both functional and
                subclassed models.
            quantization: ``None`` (default), ``"int8"``, ``"int4"`` or
                ``"fp8"`` (or a :class:`~kerasformers.quantization.\
QuantizationConfig` / scheme). When set, the model is quantized weight-only via
                :func:`kerasformers.quantization.quantize_model`: Dense/Embedding
                weights become int8 / float8-e4m3 (~4x) or block-wise packed
                int4 (~8x), and the float weights are freed. fp8 is torch/jax
                only.
            low_memory: If ``True`` **and** ``quantization`` is set, load weights
                straight into int storage without ever building the full float
                model (the no-float skeleton path,
                :func:`~kerasformers.quantization.quantize_and_load`) — so a
                checkpoint larger than device memory can be loaded quantized.
                Only applies to subclassed models whose converter assigns through
                ``model.weights`` (the standard LLM pattern); it falls back to the
                load-then-quantize path automatically for anything else.
            load_dtype: ``None`` (default, fp32) or a dtype string such as
                ``"bfloat16"`` / ``"float16"``. Builds the device model under
                that global dtype policy, so a bf16 checkpoint loads at its
                native ~2 bytes/param instead of being upcast to fp32 (≈half the
                device memory, cosine ~0.9998 vs fp32). The streamed checkpoint
                is cast to it on assign. Independent of ``quantization``, which
                runs after the build.
            **kwargs: Forwarded to the model constructor (or to
                ``from_hf`` when applicable).

        Returns:
            An initialized model instance.
        """
        if attn_implementation is not None:
            if attn_implementation not in base_attention.VALID_ATTN_IMPL:
                raise ValueError(
                    f"attn_implementation must be one of "
                    f"{base_attention.VALID_ATTN_IMPL}, got {attn_implementation!r}"
                )
            base_attention.ATTN_IMPLEMENTATION = attn_implementation
        # Build (and transfer) under the requested dtype policy so the device
        # model is allocated in e.g. bf16; post-hoc quantize runs outside it.
        with build_dtype_scope(load_dtype):
            if identifier.startswith(_HF_PREFIX):
                hf_id = identifier[len(_HF_PREFIX) :]
                model = cls.from_hf(
                    hf_id,
                    load_weights=load_weights,
                    skip_mismatch=skip_mismatch,
                    quantization=quantization,
                    low_memory=low_memory,
                    **kwargs,
                )
            else:
                model = cls.from_release(
                    identifier,
                    load_weights=load_weights,
                    skip_mismatch=skip_mismatch,
                    quantization=quantization,
                    low_memory=low_memory,
                    **kwargs,
                )
        # A no-float load already quantized in place (and recorded the config);
        # only quantize here when that path didn't run (functional models, the
        # release-`.h5` / timm paths, or low_memory=False).
        if (
            quantization is not None
            and getattr(model, "_quantization_config", None) is None
        ):
            from kerasformers.quantization import quantize_model

            model = quantize_model(model, quantization)
        return model

    @classmethod
    def _quantized_transfer(cls, model, state_dict, quantization, low_memory):
        """Apply ``cls.transfer_from_hf``; stream into int storage when asked.

        With ``low_memory`` + ``quantization`` on an unbuilt subclassed model,
        weights are quantized as they transfer (no full float model is ever
        built). Otherwise a plain float transfer runs and the caller quantizes
        afterwards. Returns ``True`` if the no-float path quantized in place.
        """
        from kerasformers.base import SubclassedBaseModel

        if (
            quantization is not None
            and low_memory
            and isinstance(model, SubclassedBaseModel)
            and not model.built
        ):
            from kerasformers.quantization import quantize_and_load

            quantize_and_load(model, quantization, cls.transfer_from_hf, state_dict)
            return True
        cls.transfer_from_hf(model, state_dict)
        return False

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

    @staticmethod
    def load_weights_from_url(model, url, skip_mismatch=False):
        """Download release weights into an (already built) ``model``.

        Handles a single ``.weights.h5`` or a sharded ``.weights.json`` index
        (downloads each shard listed in ``weight_map`` from the same release).
        """
        if url.lower().endswith(".json"):
            json_path = download_file(url)
            with open(json_path, "r") as f:
                index = json.load(f)
            if "weight_map" not in index:
                raise ValueError(
                    f"Sharded weights index '{url}' must contain 'weight_map'."
                )
            base_url = "/".join(url.split("/")[:-1])
            # weight_map values are a shard filename (older keras) or a list of
            # shard filenames per weight group (keras >= 3.14).
            shard_files = set()
            for value in index["weight_map"].values():
                if isinstance(value, list):
                    shard_files.update(value)
                else:
                    shard_files.add(value)
            for shard_file in sorted(shard_files):
                download_file(f"{base_url}/{shard_file}")
            model.load_weights(json_path, skip_mismatch=skip_mismatch)
        else:
            model.load_weights(download_file(url), skip_mismatch=skip_mismatch)

    @classmethod
    def from_release(
        cls,
        variant,
        load_weights=True,
        skip_mismatch=False,
        quantization=None,
        low_memory=False,
        **kwargs,
    ):
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
                use_safetensors = entry.get("safetensors", False)
            else:
                hf_id = None
                gated = False
                url = entry
                use_safetensors = False

            if hf_id and use_safetensors:
                # Read raw safetensors and run the model's hand-mapped transfer
                # (the same path as `hf:`): lighter than instantiating the HF
                # model, gives the exact checkpoint key layout the transfer
                # expects, and handles bf16 -> float32. Used by the Qwen
                # families, whose converters key off raw checkpoint tensors.
                state_dict = download_hf_state_dict(hf_id)
                with skip_mismatched_weights(skip_mismatch) as skipped:
                    cls._quantized_transfer(model, state_dict, quantization, low_memory)
                warn_skipped(skipped)
            elif hf_id:
                from kerasformers.conversion.hf_download_utils import (
                    load_and_convert_from_hf,
                )

                with skip_mismatched_weights(skip_mismatch) as skipped:
                    load_and_convert_from_hf(
                        model=model,
                        model_name=variant,
                        hf_model_id=hf_id,
                        transfer_fn=cls.transfer_from_hf,
                        is_gated=gated,
                    )
                warn_skipped(skipped)
            elif url:
                cls.load_weights_from_url(model, url, skip_mismatch)
            else:
                raise ValueError(
                    f"Release weights entry for variant '{variant}' has "
                    f"neither 'url' nor 'hf_id'."
                )

        return model

    @classmethod
    def from_hf(
        cls,
        hf_id,
        load_weights=True,
        variant=None,
        skip_mismatch=False,
        quantization=None,
        low_memory=False,
        **kwargs,
    ):
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
                with skip_mismatched_weights(skip_mismatch) as skipped:
                    cls.transfer_from_timm(model, state_dict)
                warn_skipped(skipped)
            return model

        with open(hf_hub_download(hf_id, "config.json"), "r") as f:
            hf_config = json.load(f)
        cls.assert_hf_model_type(hf_id, hf_config)
        kerasformers_kwargs = cls.config_from_hf(hf_config)
        kerasformers_kwargs.update(kwargs)
        model = cls(**kerasformers_kwargs)
        if load_weights:
            state_dict = download_hf_state_dict(hf_id)
            with skip_mismatched_weights(skip_mismatch) as skipped:
                cls._quantized_transfer(model, state_dict, quantization, low_memory)
            warn_skipped(skipped)
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


class PreprocessorMixin(keras.layers.Layer):
    """Single base for every kerasformers preprocessing layer — tokenizers,
    processors, image processors, and feature extractors all inherit it.

    Preprocessing layers are stateless utility layers (no weights to build) that
    take *Python* inputs — strings, chat-message lists, raw images, raw audio —
    not tensors. ``__call__`` forwards straight to ``call`` so those inputs can be
    passed positionally (Keras's ``Layer.__call__`` rejects non-tensor positional
    args).

    The loading API — ``from_weights`` / ``from_release`` / ``from_hf`` — mirrors
    the model-side :class:`WeightLoadingMixin`, so a preprocessor loads with the
    *same* identifier as its model and can pull its files from a kerasformers
    release (a variant id) or from the HF Hub (an ``"hf:org/repo"`` id)::

        gen = Qwen2Generate.from_weights("qwen2-7b-instruct")
        tok = Qwen2Tokenizer.from_weights("qwen2-7b-instruct")
        tok = CLIPTokenizer.from_weights("hf:openai/clip-vit-base-patch16")

    Subclasses (:class:`BaseTokenizer`, :class:`BaseProcessor`,
    :class:`BaseImageProcessor`, :class:`BaseAudioFeatureExtractor`) implement
    ``call`` and add their own state / ``get_config`` — the base bakes in no
    defaults.
    """

    @classmethod
    def from_weights(cls, identifier, **kwargs):
        if identifier.startswith("hf:"):
            repo = identifier[len("hf:") :]
            if "/" not in repo:
                raise ValueError(
                    f"{cls.__name__}.from_weights('hf:{repo}'): the 'hf:' prefix "
                    f"expects a Hugging Face repo id of the form 'org/name' (e.g. "
                    f"'hf:openai/clip-vit-base-patch16'), but got {repo!r} with no "
                    f"'/'. If {repo!r} is a kerasformers release variant, drop the "
                    f"'hf:' prefix: {cls.__name__}.from_weights({repo!r})."
                )
            return cls.from_hf(repo, **kwargs)
        return cls.from_release(identifier, **kwargs)

    @classmethod
    def from_release(cls, variant, /, **kwargs):
        params = inspect.signature(cls).parameters
        if "variant" in params and "variant" not in kwargs:
            kwargs["variant"] = variant
        elif (
            "hf_id" in params
            and "hf_id" not in kwargs
            and "tokenizer_file" not in kwargs
        ):
            # Gated preprocessors take `hf_id`, not `variant`; map the release
            # variant to its gated Hub repo so `from_weights(variant)` works like
            # the model's own `from_weights(variant)`.
            hf_id = cls.release_variant_hf_id(variant)
            if hf_id is not None:
                kwargs["hf_id"] = hf_id
        return cls(**kwargs)

    @classmethod
    def release_variant_hf_id(cls, variant):
        # Look up the gated Hub repo for `variant` from the model's sibling
        # `config` module: scan for any `*_WEIGHTS_URLS` dict and return the
        # variant's `hf_id` (None -> the constructor raises "needs hf_id" as usual).
        import importlib

        package = cls.__module__.rsplit(".", 1)[0]
        try:
            config = importlib.import_module(f"{package}.config")
        except ModuleNotFoundError:
            return None
        for name in dir(config):
            if name.endswith("_WEIGHTS_URLS"):
                entry = getattr(config, name).get(variant)
                if isinstance(entry, dict) and entry.get("hf_id"):
                    return entry["hf_id"]
        return None

    @classmethod
    def from_hf(cls, repo, **kwargs):
        if "hf_id" not in inspect.signature(cls).parameters:
            raise NotImplementedError(
                f"{cls.__name__} cannot load from an 'hf:' repo — its constructor "
                f"takes no `hf_id`. Use a release variant, or override `from_hf` "
                f"to fetch the files from {repo!r}."
            )
        return cls(hf_id=repo, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must implement `call`.")
