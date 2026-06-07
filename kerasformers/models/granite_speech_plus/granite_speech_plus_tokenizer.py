import keras

from kerasformers.models.granite_speech.granite_speech_tokenizer import (
    GraniteSpeechTokenizer,
)

from .config import GRANITE_SPEECH_PLUS_TOKENIZER_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechPlusTokenizer(GraniteSpeechTokenizer):
    """Granite Speech 4.1-plus tokenizer (granite-4.0 BPE, 100353-token vocab).

    Identical to GraniteSpeechTokenizer except the bundled release file defaults
    to the plus tokenizer.json instead of the 3.3-2b one; audio/eos ids are read
    from the loaded vocab.
    """

    TOKENIZER_URL = GRANITE_SPEECH_PLUS_TOKENIZER_URL
