import keras

from kerasformers.models.deberta_v2.deberta_v2_tokenizer import DebertaV2Tokenizer

from .deberta_v3_config import DEBERTA_V3_TOKENIZER_URLS


@keras.saving.register_keras_serializable(package="kerasformers")
class DebertaV3Tokenizer(DebertaV2Tokenizer):
    """DeBERTa-v3 SentencePiece tokenizer.

    Identical machinery to :class:`DebertaV2Tokenizer`; only the per-variant
    ``tokenizer.json`` (a different SentencePiece vocab) differs.
    """

    TOKENIZER_URLS = DEBERTA_V3_TOKENIZER_URLS
    DEFAULT_VARIANT = "deberta_v3_base"
