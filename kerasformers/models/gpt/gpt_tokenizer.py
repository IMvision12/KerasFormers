import keras
import numpy as np

from kerasformers.base import BaseTokenizer
from kerasformers.conversion import download_file

from .config import GPT_MERGES_URL, GPT_VOCAB_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class GptTokenizer(BaseTokenizer):
    """Original GPT BPE tokenizer (``tokenizers`` backend).

    Builds a ``tokenizers.Tokenizer`` matching Hugging Face's original GPT: NFC +
    lowercase normalization, BERT-style pre-tokenization, and byte-pair encoding
    over ``vocab.json`` + ``merges.txt`` with ``</w>`` word-boundary suffixes.
    Exposes ``encode`` / ``decode`` and a ``call`` that tokenizes text(s) into
    padded ``{"input_ids", "attention_mask"}``. GPT is a base LM with no special
    end-of-text token.

    Args:
        vocab_file: Path to ``vocab.json``. When ``None`` (and ``merges_file`` is
            also ``None``), downloads the default kerasformers-release files on
            first use.
        merges_file: Path to ``merges.txt``. See ``vocab_file``.
        unk_token: Unknown-token string (default ``"<unk>"``).
    """

    def __init__(self, vocab_file=None, merges_file=None, unk_token="<unk>", **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
        from tokenizers.models import BPE

        if vocab_file is None and merges_file is None:
            vocab_file = download_file(GPT_VOCAB_URL)
            merges_file = download_file(GPT_MERGES_URL)
        elif (vocab_file is None) != (merges_file is None):
            missing = "merges_file" if merges_file is None else "vocab_file"
            provided = "vocab_file" if merges_file is None else "merges_file"
            raise ValueError(
                f"GptTokenizer requires both vocab_file (vocab.json) and "
                f"merges_file (merges.txt), but only {provided} was provided. "
                f"Either supply {missing} as well, or omit both to download the "
                f"default kerasformers-release files automatically."
            )
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.unk_token = unk_token

        tok = Tokenizer(
            BPE(
                vocab=vocab_file,
                merges=merges_file,
                unk_token=unk_token,
                end_of_word_suffix="</w>",
                fuse_unk=False,
            )
        )
        if tok.token_to_id(unk_token) is not None:
            tok.add_special_tokens([unk_token])
        tok.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Lowercase()]
        )
        tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tok.decoder = decoders.BPEDecoder(suffix="</w>")
        self._tok = tok
        self.eos_token_id = None

    @classmethod
    def from_hf(cls, repo, **kwargs):
        """Load a GPT finetune's ``vocab.json`` + ``merges.txt`` from the HF
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
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        ids = [self.encode(t) for t in texts]
        max_len = max(len(s) for s in ids)
        input_ids = np.zeros((len(ids), max_len), dtype="int32")
        attention_mask = np.zeros((len(ids), max_len), dtype="int32")
        for i, s in enumerate(ids):
            input_ids[i, : len(s)] = s
            attention_mask[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

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
                "vocab_file": self.vocab_file,
                "merges_file": self.merges_file,
                "unk_token": self.unk_token,
            }
        )
        return config
