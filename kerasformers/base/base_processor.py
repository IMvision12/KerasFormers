from kerasformers.base.base_mixin import PreprocessorMixin


class BaseProcessor(PreprocessorMixin):
    """Base class for kerasformers multi-modal processors.

    Multi-modal processors compose a :class:`BaseTokenizer` and a
    :class:`BaseImageProcessor` / :class:`BaseAudioFeatureExtractor` into one
    callable. Subclasses set ``self.tokenizer`` / ``self.image_processor`` /
    ``self.feature_extractor`` in ``__init__`` and implement ``call`` to dispatch
    over their component(s); ``decode`` / ``batch_decode`` are wired through to
    the tokenizer. The loading API and the ``__call__`` -> ``call`` forwarder are
    inherited from :class:`PreprocessorMixin`.
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
