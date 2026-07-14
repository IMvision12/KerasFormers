import keras
from tokenizers import Tokenizer

from kerasformers.base import BaseTokenizer

from .gpt2_config import GPT2_TOKENIZER_URLS


@keras.saving.register_keras_serializable(package="kerasformers")
class GPT2Tokenizer(BaseTokenizer):
    """GPT-2 byte-level BPE tokenizer (``tokenizers`` backend).

    Loads the HuggingFace fast-tokenizer ``tokenizer.json`` for ``variant`` from the
    ``gpt`` release tag (or an explicit ``tokenizer_file``). ``<|endoftext|>`` is the
    single special token; ``call`` pads batches to the longest sequence. GPT-2 is a
    base LM with no chat template.

    Args:
        variant: GPT-2 variant key (default ``"gpt2"``).
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides variant).
        eos_token: End-of-text token string (default ``"<|endoftext|>"``).
    """

    TOKENIZER_URLS = GPT2_TOKENIZER_URLS
    DEFAULT_VARIANT = "gpt2"

    def __init__(
        self, variant=None, tokenizer_file=None, eos_token="<|endoftext|>", **kwargs
    ):
        super().__init__(**kwargs)
        self.variant = variant or self.DEFAULT_VARIANT
        tokenizer_file = self.resolve_tokenizer_json(self.variant, tokenizer_file)
        self.tokenizer_file = tokenizer_file
        self.eos_token = eos_token
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.eos_token_id = self._tok.token_to_id(eos_token)

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
                "eos_token": self.eos_token,
            }
        )
        return config
