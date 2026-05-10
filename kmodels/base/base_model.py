import keras

from kmodels.weight_utils import download_file

_HF_PREFIX = "hf:"


class BaseModel(keras.Model):
    """Base class for kmodels models with unified weight loading.

    Subclasses are Functional Keras models that share a single entry
    point for loading pretrained weights, regardless of source. Two
    sources are supported by default:

    1. **kmodels release** — weights hosted on GitHub Releases keyed by
       a short variant string (e.g. ``"owlvit-base-patch32"``).
    2. **HuggingFace** — weights pulled from a HF Hub repo, identified
       by an ``"hf:org/repo"`` string. Works for original HF checkpoints
       and for community fine-tunes that share the same architecture.

    To wire a new model in, a subclass sets three class attributes and
    optionally overrides two hooks:

    .. code-block:: python

        class OwlViTDetect(BaseModel):
            KMODELS_CONFIG = OWLVIT_CONFIG          # variant -> kwargs
            KMODELS_WEIGHTS = OWLVIT_WEIGHTS        # variant -> url
            HF_MODEL_CLS = transformers.OwlViTForObjectDetection

            @classmethod
            def _config_from_hf(cls, hf_config): ...

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
    HF_MODEL_CLS = None

    @classmethod
    def from_weights(cls, identifier, load_weights=True, **kwargs):
        """Build a model and (optionally) load pretrained weights.

        Args:
            identifier: Either a kmodels variant string (e.g.
                ``"owlvit-base-patch32"``) which resolves against
                ``cls.KMODELS_CONFIG`` / ``cls.KMODELS_WEIGHTS``, or an
                ``"hf:<org>/<repo>"`` string which pulls config and
                weights from HuggingFace via ``cls.HF_MODEL_CLS``.
            load_weights: If ``False``, only the architecture is built
                (random init). The kmodels release URL is not fetched
                and HF weights are not transferred, but the HF config
                is still used to size the model when an ``"hf:"`` id
                is passed.
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
        if cls.HF_MODEL_CLS is None:
            raise NotImplementedError(
                f"{cls.__name__} must set HF_MODEL_CLS to load from HuggingFace."
            )
        try:
            from transformers import AutoConfig
        except ImportError as e:
            raise ImportError(
                "Loading from HuggingFace requires the `transformers` package. "
                "Install it with `pip install transformers`."
            ) from e

        if load_weights:
            hf_model = cls.HF_MODEL_CLS.from_pretrained(hf_id)
            hf_config = hf_model.config
            state_dict = {
                k: v.cpu().numpy() if hasattr(v, "cpu") else v
                for k, v in hf_model.state_dict().items()
            }
        else:
            hf_config = AutoConfig.from_pretrained(hf_id)
            state_dict = None

        kmodels_kwargs = cls._config_from_hf(hf_config)
        model = cls(**kmodels_kwargs, **kwargs)

        if load_weights:
            cls._transfer_from_hf(model, state_dict)

        return model

    @classmethod
    def _config_from_hf(cls, hf_config):
        """Map a HuggingFace config to ``cls.__init__`` kwargs.

        Subclasses must override this when ``HF_MODEL_CLS`` is set.
        """
        raise NotImplementedError(f"{cls.__name__}._config_from_hf is not implemented.")

    @classmethod
    def _transfer_from_hf(cls, keras_model, hf_state_dict):
        """Transfer weights from an HF ``state_dict`` into ``keras_model``.

        Subclasses must override this when ``HF_MODEL_CLS`` is set.
        """
        raise NotImplementedError(
            f"{cls.__name__}._transfer_from_hf is not implemented."
        )
