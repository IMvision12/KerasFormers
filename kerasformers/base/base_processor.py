import keras

from kerasformers.base.base_mixin import PreloadMixin


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
