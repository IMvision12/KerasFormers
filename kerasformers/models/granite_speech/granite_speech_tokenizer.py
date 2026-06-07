import keras

from kerasformers.base import BaseTokenizer
from kerasformers.conversion import download_file

from .config import GRANITE_SPEECH_TOKENIZER_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechTokenizer(BaseTokenizer):
    """Granite BPE tokenizer (``tokenizers`` backend) with the ``<|audio|>`` token.

    Args:
        tokenizer_file: Path to ``tokenizer.json``. When ``None``, downloads the
            bundled kerasformers-release file.
        audio_token: The audio placeholder token string.
    """

    TOKENIZER_URL = GRANITE_SPEECH_TOKENIZER_URL

    def __init__(
        self,
        tokenizer_file=None,
        audio_token="<|audio|>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        from tokenizers import AddedToken, Tokenizer

        if tokenizer_file is None:
            tokenizer_file = download_file(self.TOKENIZER_URL)
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)

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
        """Load a Granite Speech repo's ``tokenizer.json`` from the HF ``repo``
        instead of the bundled kerasformers-release default."""
        from huggingface_hub import hf_hub_download

        return cls(tokenizer_file=hf_hub_download(repo, "tokenizer.json"), **kwargs)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def call(self, inputs):
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        ids = [self.encode(t) for t in texts]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self._tok.decode(
            [int(i) for i in ids], skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer_file": self.tokenizer_file,
                "audio_token": self.audio_token,
            }
        )
        return config
