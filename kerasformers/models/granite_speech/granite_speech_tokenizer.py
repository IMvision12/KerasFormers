import os

import keras

from kerasformers.base import BaseTokenizer
from kerasformers.conversion import download_file

from .config import (
    GRANITE_SPEECH_MERGES_URL,
    GRANITE_SPEECH_SPECIAL_TOKENS,
    GRANITE_SPEECH_VOCAB_URL,
)


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechTokenizer(BaseTokenizer):
    """Granite byte-level BPE tokenizer (``tokenizers`` backend) with ``<|audio|>``.

    Built from ``vocab.json`` + ``merges.txt`` (ByteLevel BPE pipeline) with the
    Granite special tokens registered on top, so ``<|audio|>`` / ``<|end_of_text|>``
    resolve to their checkpoint ids. The files are pulled from the
    ``granite_speech`` release tag on ``github.com/IMvision12/KerasFormers`` unless
    explicit paths are given.

    Args:
        vocab_file / merges_file: Optional explicit paths. When ``None``, the
            bundled kerasformers-release files are downloaded.
        audio_token: The audio placeholder token string.
    """

    VOCAB_URL = GRANITE_SPEECH_VOCAB_URL
    MERGES_URL = GRANITE_SPEECH_MERGES_URL
    SPECIAL_TOKENS = GRANITE_SPEECH_SPECIAL_TOKENS

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        audio_token="<|audio|>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        from tokenizers import AddedToken, Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel

        if vocab_file is None or not os.path.exists(vocab_file):
            vocab_file = download_file(self.VOCAB_URL)
        if merges_file is None or not os.path.exists(merges_file):
            merges_file = download_file(self.MERGES_URL)
        self.vocab_file = vocab_file
        self.merges_file = merges_file

        tok = Tokenizer(BPE(vocab=vocab_file, merges=merges_file, fuse_unk=False))
        tok.pre_tokenizer = ByteLevel(
            add_prefix_space=False, trim_offsets=True, use_regex=True
        )
        tok.decoder = ByteLevelDecoder()
        tok.add_special_tokens(
            [AddedToken(t, special=True, normalized=False) for t in self.SPECIAL_TOKENS]
        )
        self._tok = tok
        self.register_special_ids(audio_token)

    def register_special_ids(self, audio_token):
        from tokenizers import AddedToken

        self.audio_token = audio_token
        if self._tok.token_to_id(audio_token) is None:
            self._tok.add_special_tokens(
                [AddedToken(audio_token, special=True, normalized=False)]
            )
        self.audio_token_id = self._tok.token_to_id(audio_token)
        self.eos_token = "<|end_of_text|>"
        eos_id = self._tok.token_to_id(self.eos_token)
        self.eos_token_id = eos_id if eos_id is not None else 0

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        return cls(
            vocab_file=hf_hub_download(repo, "vocab.json"),
            merges_file=hf_hub_download(repo, "merges.txt"),
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        return {"input_ids": [self.encode(t) for t in texts]}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_file": self.vocab_file,
                "merges_file": self.merges_file,
                "audio_token": self.audio_token,
            }
        )
        return config
