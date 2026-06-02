import inspect

import keras


class PreloadMixin:
    """Shared ``from_weights`` / ``from_release`` / ``from_hf`` for kerasformers
    preprocessing layers (tokenizers, processors, image processors, feature
    extractors).

    Mirrors the model-side loading API so a preprocessor loads with the *same*
    identifier as its model::

        gen = Qwen2Generate.from_weights("qwen2-7b-instruct")
        tok = Qwen2Tokenizer.from_weights("qwen2-7b-instruct")

    Kept as a plain mixin (not a ``keras.Layer``) so it composes onto any
    preprocessing base without touching the layer MRO — same spirit as
    :class:`WeightLoadingMixin` on the model side.
    """

    @classmethod
    def from_weights(cls, identifier, **kwargs):
        if identifier.startswith("hf:"):
            return cls.from_hf(identifier[len("hf:") :], **kwargs)
        return cls.from_release(identifier, **kwargs)

    @classmethod
    def from_release(cls, variant, /, **kwargs):
        # The official preprocessor is shared across a family's sizes, so the
        # default just builds the class default (its constructor pulls the bundled
        # / hub assets). Override only for per-variant resolution.
        return cls(**kwargs)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        # Default: download via the ``hf_id`` constructor arg (e.g. tokenizer.json).
        # Families assembled from multiple files (e.g. CLIP's vocab.json +
        # merges.txt) override this to fetch them from ``repo``.
        if "hf_id" not in inspect.signature(cls).parameters:
            raise NotImplementedError(
                f"{cls.__name__} cannot load from an 'hf:' repo — its constructor "
                f"takes no `hf_id`. Use a release variant, or override `from_hf` "
                f"to fetch the files from {repo!r}."
            )
        return cls(hf_id=repo, **kwargs)


class BasePreprocessingLayer(PreloadMixin, keras.layers.Layer):
    """Base for every kerasformers preprocessing layer.

    Preprocessing layers are stateless utility layers (no weights to build) that
    take *Python* inputs — strings, chat-message lists, raw images, raw audio —
    not tensors. ``__call__`` forwards straight to ``call`` so those inputs can be
    passed positionally (Keras's ``Layer.__call__`` rejects non-tensor positional
    args). Loading (``from_weights`` / ``from_release`` / ``from_hf``) is inherited
    from :class:`PreloadMixin`.

    Subclasses (``BaseTokenizer``, ``BaseProcessor``, ``BaseImageProcessor``,
    ``BaseAudioFeatureExtractor``) implement ``call`` and add their own state /
    ``get_config`` — the base bakes in no defaults.
    """

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must implement `call`.")


class BaseProcessor(BasePreprocessingLayer):
    """Base class for kerasformers multi-modal processors.

    Multi-modal processors compose a :class:`BaseTokenizer` and a
    :class:`BaseImageProcessor` / :class:`BaseAudioFeatureExtractor` into one
    callable. Subclasses set ``self.tokenizer`` / ``self.image_processor`` /
    ``self.feature_extractor`` in ``__init__`` and implement ``call`` to dispatch
    over their component(s); ``decode`` / ``batch_decode`` are wired through to
    the tokenizer. The loading API and the ``__call__`` -> ``call`` forwarder are
    inherited from :class:`BasePreprocessingLayer`.
    """

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
