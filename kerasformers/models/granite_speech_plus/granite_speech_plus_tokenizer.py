import os

import keras

from kerasformers.base import BaseTokenizer
from kerasformers.conversion import download_file
from kerasformers.models.granite_speech.granite_speech_tokenizer import (
    GraniteSpeechTokenizer,
)

from .config import GRANITE_SPEECH_PLUS_TOKENIZER_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechPlusTokenizer(GraniteSpeechTokenizer):
    """Granite Speech 4.1-plus tokenizer (granite-4.0 BPE, 100353-token vocab).

    The plus checkpoint does not publish ``vocab.json`` / ``merges.txt``, so this
    loads the combined ``tokenizer.json`` (from the ``granite_speech`` release tag)
    rather than rebuilding from vocab + merges. The audio/eos id setup and the
    encode/decode pipeline are shared with :class:`GraniteSpeechTokenizer`.
    """

    TOKENIZER_URL = GRANITE_SPEECH_PLUS_TOKENIZER_URL

    def __init__(self, tokenizer_file=None, audio_token="<|audio|>", **kwargs):
        # plus loads a combined tokenizer.json, not vocab+merges, so skip the
        # GraniteSpeechTokenizer constructor and build the backend directly.
        BaseTokenizer.__init__(self, **kwargs)
        from tokenizers import Tokenizer

        if tokenizer_file is None or not os.path.exists(tokenizer_file):
            tokenizer_file = download_file(self.TOKENIZER_URL)
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.register_special_ids(audio_token)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        return cls(tokenizer_file=hf_hub_download(repo, "tokenizer.json"), **kwargs)

    def get_config(self):
        config = BaseTokenizer.get_config(self)
        config.update(
            {
                "tokenizer_file": self.tokenizer_file,
                "audio_token": self.audio_token,
            }
        )
        return config
