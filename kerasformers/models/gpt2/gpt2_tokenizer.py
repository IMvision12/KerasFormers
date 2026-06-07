import keras

from kerasformers.base import BaseTokenizer
from kerasformers.conversion import download_file

from .config import GPT2_MERGES_URL, GPT2_VOCAB_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class GPT2Tokenizer(BaseTokenizer):
    """GPT-2 byte-level BPE tokenizer (``tokenizers`` backend).

    Builds a ``tokenizers.Tokenizer`` matching Hugging Face GPT-2: byte-level
    pre-tokenization and BPE over ``vocab.json`` + ``merges.txt`` (no
    normalization), with ``<|endoftext|>`` as the single special token. Exposes
    ``encode`` / ``decode`` and a ``call`` that tokenizes text(s) into padded
    ``{"input_ids", "attention_mask"}`` ready for ``model.generate``. GPT-2 is a
    base LM with no chat template.

    Args:
        vocab_file: Path to ``vocab.json``. When ``None`` (and ``merges_file`` is
            also ``None``), downloads the default kerasformers-release files on
            first use.
        merges_file: Path to ``merges.txt``. See ``vocab_file``.
        eos_token: End-of-text token string (default ``"<|endoftext|>"``).
    """

    def __init__(
        self, vocab_file=None, merges_file=None, eos_token="<|endoftext|>", **kwargs
    ):
        super().__init__(**kwargs)
        from tokenizers import AddedToken, Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import ByteLevel as ByteLevelProcessor

        if vocab_file is None and merges_file is None:
            vocab_file = download_file(GPT2_VOCAB_URL)
            merges_file = download_file(GPT2_MERGES_URL)
        elif (vocab_file is None) != (merges_file is None):
            missing = "merges_file" if merges_file is None else "vocab_file"
            provided = "vocab_file" if merges_file is None else "merges_file"
            raise ValueError(
                f"GPT2Tokenizer requires both vocab_file (vocab.json) and "
                f"merges_file (merges.txt), but only {provided} was provided. "
                f"Either supply {missing} as well, or omit both to download the "
                f"default kerasformers-release files automatically."
            )
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.eos_token = eos_token

        tok = Tokenizer(
            BPE(
                vocab=vocab_file,
                merges=merges_file,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        tok.post_processor = ByteLevelProcessor(trim_offsets=False)
        tok.add_special_tokens([AddedToken(eos_token, special=True, normalized=False)])
        self._tok = tok
        self.eos_token_id = self._tok.token_to_id(eos_token)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        """Load a GPT-2 finetune's ``vocab.json`` + ``merges.txt`` from the HF
        ``repo`` instead of the bundled kerasformers-release default."""
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
                "vocab_file": self.vocab_file,
                "merges_file": self.merges_file,
                "eos_token": self.eos_token,
            }
        )
        return config
