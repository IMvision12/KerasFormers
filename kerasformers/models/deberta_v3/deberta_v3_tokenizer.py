import keras

from kerasformers.models.deberta_v2.deberta_v2_tokenizer import DebertaV2Tokenizer
from kerasformers.weight_utils import download_file

from .config import DEBERTA_V3_VOCAB_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class DebertaV3Tokenizer(DebertaV2Tokenizer):
    """DeBERTa-v3 SentencePiece tokenizer.

    Identical machinery to :class:`DebertaV2Tokenizer` (SentencePiece, no fairseq
    offset, ``[CLS] A [SEP]`` post-processing); only the default ``spm.model``
    differs.
    """

    def __init__(self, vocab_file=None, **kwargs):
        if vocab_file is None:
            vocab_file = download_file(DEBERTA_V3_VOCAB_URL)
        super().__init__(vocab_file=vocab_file, **kwargs)
