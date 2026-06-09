import keras

from kerasformers.models.granite_speech.granite_speech_tokenizer import (
    GraniteSpeechTokenizer,
)

from .config import GRANITE_SPEECH_PLUS_TOKENIZER_URLS


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechPlusTokenizer(GraniteSpeechTokenizer):
    """Granite Speech 4.1-plus tokenizer (granite-4.0 BPE, 100353-token vocab).

    Identical machinery to :class:`GraniteSpeechTokenizer`; only the per-variant
    ``tokenizer.json`` differs (``<|audio|>`` = 100352, eos = 100257).
    """

    TOKENIZER_URLS = GRANITE_SPEECH_PLUS_TOKENIZER_URLS
    DEFAULT_VARIANT = "granite_speech_4_1_2b_plus"
