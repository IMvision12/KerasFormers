import keras
from tokenizers import Tokenizer

from kerasformers.base import BaseTokenizer

from .gpt_config import GPT_TOKENIZER_URLS


@keras.saving.register_keras_serializable(package="kerasformers")
class GptTokenizer(BaseTokenizer):
    """Original GPT BPE tokenizer (``tokenizers`` backend).

    Loads the HuggingFace fast-tokenizer ``tokenizer.json`` for ``variant`` from the
    ``gpt`` release tag (or an explicit ``tokenizer_file``). ``call`` pads batches to
    the longest sequence. GPT is a base LM with no end-of-text token.

    Args:
        variant: GPT variant key (default ``"gpt"``).
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides variant).
        unk_token: Unknown-token string (default ``"<unk>"``).
    """

    TOKENIZER_URLS = GPT_TOKENIZER_URLS
    DEFAULT_VARIANT = "gpt"

    def __init__(self, variant=None, tokenizer_file=None, unk_token="<unk>", **kwargs):
        super().__init__(**kwargs)
        self.variant = variant or self.DEFAULT_VARIANT
        tokenizer_file = self.resolve_tokenizer_json(self.variant, tokenizer_file)
        self.tokenizer_file = tokenizer_file
        self.unk_token = unk_token
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.eos_token_id = None

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        return cls(tokenizer_file=hf_hub_download(repo, "tokenizer.json"), **kwargs)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        input_ids, attention_mask = self.pad_batch([self.encode(t) for t in texts])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "tokenizer_file": self.tokenizer_file,
                "unk_token": self.unk_token,
            }
        )
        return config
