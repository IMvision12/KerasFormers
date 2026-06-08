import keras

from kerasformers.base.base_mixin import PreprocessorMixin


class BaseProcessor(PreprocessorMixin):
    """Base class for kerasformers multi-modal processors.

    A processor **composes** (has-a) a :class:`BaseTokenizer` and a
    :class:`BaseImageProcessor` / :class:`BaseAudioFeatureExtractor`; it is not a
    subclass of them. Each subclass declares the component classes it uses via the
    ``TOKENIZER_CLS`` / ``IMAGE_PROCESSOR_CLS`` / ``FEATURE_EXTRACTOR_CLS`` class
    attributes and stores the built instances on ``self.tokenizer`` /
    ``self.image_processor`` / ``self.feature_extractor``. ``__init__`` accepts
    pre-built components (used by the loaders) or builds them from kwargs.

    The base then provides, generically over whatever components are declared:

    * ``from_hf(repo)`` — loads **every** component from the HF ``repo`` (tokenizer
      files + image processor / feature extractor), so ``from_weights("hf:org/repo")``
      returns a complete processor.
    * ``get_config`` / ``from_config`` — serialize/deserialize the components.
    * ``decode`` / ``batch_decode`` — wired through to ``self.tokenizer``.

    Subclasses implement ``call`` (the modality dispatch) and, if they carry extra
    scalar state, extend ``get_config``. The loading API + ``__call__`` -> ``call``
    forwarder are inherited from :class:`PreprocessorMixin`.
    """

    TOKENIZER_CLS = None
    IMAGE_PROCESSOR_CLS = None
    FEATURE_EXTRACTOR_CLS = None
    COMPONENTS = ("tokenizer", "image_processor", "feature_extractor")

    @classmethod
    def from_hf(cls, repo, **kwargs):
        parts = {}
        if cls.TOKENIZER_CLS is not None:
            parts["tokenizer"] = cls.TOKENIZER_CLS.from_hf(repo)
        if cls.IMAGE_PROCESSOR_CLS is not None:
            parts["image_processor"] = cls.IMAGE_PROCESSOR_CLS.from_hf(repo)
        if cls.FEATURE_EXTRACTOR_CLS is not None:
            parts["feature_extractor"] = cls.FEATURE_EXTRACTOR_CLS.from_hf(repo)
        return cls(**parts, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise AttributeError(
                f"{type(self).__name__}.decode() requires `self.tokenizer` to be set."
            )
        return tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise AttributeError(
                f"{type(self).__name__}.batch_decode() requires "
                "`self.tokenizer` to be set."
            )
        return tokenizer.batch_decode(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        for attr in self.COMPONENTS:
            component = getattr(self, attr, None)
            if component is not None:
                config[attr] = keras.saving.serialize_keras_object(component)
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        for attr in cls.COMPONENTS:
            if isinstance(config.get(attr), dict):
                config[attr] = keras.saving.deserialize_keras_object(config[attr])
        return cls(**config)
