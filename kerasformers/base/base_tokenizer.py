from kerasformers.base.base_processor import BasePreprocessingLayer


class BaseTokenizer(BasePreprocessingLayer):
    """Abstract base for kerasformers tokenizers.

    Subclasses implement ``call`` (text -> ids) and ``decode`` (ids -> text);
    ``batch_decode`` is a pure-Python loop over ``decode``. The loading API
    (``from_weights`` / ``from_release`` / ``from_hf``) and the ``__call__`` ->
    ``call`` forwarder are inherited from :class:`BasePreprocessingLayer`.

    Concrete tokenizers add their own state (vocab path, merges, special-token
    ids, BPE / SentencePiece backend) and ``get_config`` payload — the base
    intentionally bakes in no defaults.
    """

    def call(self, inputs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `call(inputs)`."
        )

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement `decode(ids, skip_special_tokens)`."
        )

    def batch_decode(self, ids_batch, skip_special_tokens: bool = True):
        return [self.decode(ids, skip_special_tokens) for ids in ids_batch]
