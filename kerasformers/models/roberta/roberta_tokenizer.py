import json
from typing import List, Union

import keras
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import RobertaProcessing

from kerasformers.base import BaseTokenizer
from kerasformers.conversion import download_file

from .config import ROBERTA_MERGES_URL, ROBERTA_VOCAB_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class RobertaTokenizer(BaseTokenizer):
    """RoBERTa byte-level BPE tokenizer, built on the `tokenizers` library (Rust).

    Assembles a `tokenizers.Tokenizer` that matches Hugging Face RoBERTa: no
    normalization, byte-level pre-tokenization, BPE over ``vocab.json`` +
    ``merges.txt``, and ``<s> A </s>`` / ``<s> A </s> </s> B </s>`` RoBERTa-style
    post-processing. ``call`` returns the ``input_ids`` / ``attention_mask`` /
    ``token_type_ids`` dict expected by :class:`RobertaModel` (token types are
    always ``0`` for RoBERTa).

    Args:
        vocab_file: Path to ``vocab.json``. When ``None`` (and ``merges_file`` is
            also ``None``), downloads the default kerasformers-release files on
            first use.
        merges_file: Path to ``merges.txt``. See ``vocab_file``.
        max_seq_len: Truncation length (default 512). Batches are padded to the
            longest sequence, so short inputs stay short.
        bos_token / eos_token / unk_token / pad_token / mask_token: Special tokens.
    """

    def __init__(
        self,
        vocab_file: str = None,
        merges_file: str = None,
        max_seq_len: int = 512,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vocab_file is None and merges_file is None:
            vocab_file = download_file(ROBERTA_VOCAB_URL)
            merges_file = download_file(ROBERTA_MERGES_URL)
        elif (vocab_file is None) != (merges_file is None):
            missing = "merges_file" if merges_file is None else "vocab_file"
            provided = "vocab_file" if merges_file is None else "merges_file"
            raise ValueError(
                f"RobertaTokenizer requires both vocab_file (vocab.json) and "
                f"merges_file (merges.txt), but only {provided} was provided. "
                f"Either supply {missing} as well, or omit both to download the "
                f"default kerasformers-release files automatically."
            )
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.max_seq_len = max_seq_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.bos_token_id = self.encoder[bos_token]
        self.eos_token_id = self.encoder[eos_token]
        self.unk_token_id = self.encoder[unk_token]
        self.pad_token_id = self.encoder[pad_token]
        self.mask_token_id = self.encoder[mask_token]

        tok = Tokenizer(
            BPE(
                vocab=vocab_file,
                merges=merges_file,
                unk_token=unk_token,
                fuse_unk=False,
            )
        )
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=True)
        tok.post_processor = RobertaProcessing(
            sep=(eos_token, self.eos_token_id),
            cls=(bos_token, self.bos_token_id),
            trim_offsets=True,
            add_prefix_space=False,
        )
        tok.decoder = ByteLevelDecoder()
        tok.add_special_tokens(
            [
                AddedToken(bos_token, special=True, normalized=False),
                AddedToken(pad_token, special=True, normalized=False),
                AddedToken(eos_token, special=True, normalized=False),
                AddedToken(unk_token, special=True, normalized=False),
                AddedToken(mask_token, special=True, normalized=False, lstrip=True),
            ]
        )
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(pad_id=self.pad_token_id, pad_token=pad_token)
        self._tok = tok

    @classmethod
    def from_hf(cls, repo, **kwargs):
        """Load a RoBERTa finetune's ``vocab.json`` + ``merges.txt`` from the HF
        ``repo`` instead of the bundled kerasformers-release default."""
        from huggingface_hub import hf_hub_download

        return cls(
            vocab_file=hf_hub_download(repo, "vocab.json"),
            merges_file=hf_hub_download(repo, "merges.txt"),
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self._tok.encode(text, add_special_tokens=False).ids
        encs = self._tok.encode_batch(text, add_special_tokens=False)
        return [e.ids for e in encs]

    def detokenize(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def call(self, inputs: Union[str, List[str]], text_pair=None):
        return self.encode_batch_to_inputs(inputs, text_pair)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_file": self.vocab_file,
                "merges_file": self.merges_file,
                "max_seq_len": self.max_seq_len,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "mask_token": self.mask_token,
            }
        )
        return config
