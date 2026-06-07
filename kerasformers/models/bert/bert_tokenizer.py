import json
from typing import List, Union

import keras
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing

from kerasformers.base import BaseTokenizer
from kerasformers.conversion import download_file

from .config import BERT_VOCAB_CONFIG


@keras.saving.register_keras_serializable(package="kerasformers")
class BertTokenizer(BaseTokenizer):
    """BERT WordPiece tokenizer, built on the `tokenizers` library (Rust).

    Assembles a `tokenizers.Tokenizer` that matches Hugging Face BERT: the
    BERT text normalizer (clean text, handle CJK, optional lowercase + accent
    strip), whitespace/punctuation pre-tokenization, greedy WordPiece over
    ``vocab.txt``, and ``[CLS] A [SEP] B [SEP]`` template post-processing with
    segment (token-type) ids. ``call`` returns the ``input_ids`` /
    ``attention_mask`` / ``token_type_ids`` dict expected by :class:`BertModel`.

    Args:
        vocab_file: Path to ``vocab.txt`` (one token per line). When ``None``,
            downloads the default kerasformers-release vocab on first use.
        max_seq_len: Truncation length (default 512). Batches are padded to the
            longest sequence, so short inputs stay short.
        do_lower_case: Lowercase + strip accents (True for the uncased models).
        unk_token / sep_token / pad_token / cls_token / mask_token: Special tokens.
    """

    def __init__(
        self,
        vocab_file: str = None,
        max_seq_len: int = 512,
        do_lower_case: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vocab_file is None:
            vocab_file = download_file(
                BERT_VOCAB_CONFIG["bert_base_uncased"]["vocab_url"]
            )
        self.vocab_file = vocab_file
        self.max_seq_len = max_seq_len
        self.do_lower_case = do_lower_case
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}
        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}

        self.cls_token_id = self.vocab[cls_token]
        self.sep_token_id = self.vocab[sep_token]
        self.pad_token_id = self.vocab[pad_token]
        self.unk_token_id = self.vocab[unk_token]
        self.mask_token_id = self.vocab[mask_token]

        tok = Tokenizer(WordPiece(self.vocab, unk_token=unk_token))
        tok.normalizer = BertNormalizer(lowercase=do_lower_case)
        tok.pre_tokenizer = BertPreTokenizer()
        tok.post_processor = TemplateProcessing(
            single=f"{cls_token}:0 $A:0 {sep_token}:0",
            pair=f"{cls_token}:0 $A:0 {sep_token}:0 $B:1 {sep_token}:1",
            special_tokens=[
                (cls_token, self.cls_token_id),
                (sep_token, self.sep_token_id),
            ],
        )
        tok.decoder = WordPieceDecoder(prefix="##")
        # Protect special tokens from normalization/splitting so a literal
        # "[MASK]" / "[CLS]" in the input maps to its id instead of being
        # lowercased and WordPiece-split (matches Hugging Face behavior).
        tok.add_special_tokens(
            [
                AddedToken(t, special=True, normalized=False)
                for t in (unk_token, sep_token, pad_token, cls_token, mask_token)
            ]
        )
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(pad_id=self.pad_token_id, pad_token=pad_token)
        self._tok = tok

    @classmethod
    def from_release(cls, variant, /, **kwargs):
        """Build the tokenizer for a release ``variant``, resolving its vocab and
        casing (the cased models use a different vocab and ``do_lower_case=False``)."""
        entry = BERT_VOCAB_CONFIG.get(variant, {})
        if "vocab_url" in entry and "vocab_file" not in kwargs:
            kwargs["vocab_file"] = download_file(entry["vocab_url"])
        if "do_lower_case" in entry:
            kwargs.setdefault("do_lower_case", entry["do_lower_case"])
        return cls(**kwargs)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        """Load a BERT finetune's ``vocab.txt`` (and casing) from the HF ``repo``
        instead of the bundled kerasformers-release default."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        if "do_lower_case" not in kwargs:
            try:
                with open(hf_hub_download(repo, "tokenizer_config.json")) as f:
                    tok_config = json.load(f)
                if "do_lower_case" in tok_config:
                    kwargs["do_lower_case"] = tok_config["do_lower_case"]
            except EntryNotFoundError:
                pass
        return cls(vocab_file=hf_hub_download(repo, "vocab.txt"), **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

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
                "max_seq_len": self.max_seq_len,
                "do_lower_case": self.do_lower_case,
                "unk_token": self.unk_token,
                "sep_token": self.sep_token,
                "pad_token": self.pad_token,
                "cls_token": self.cls_token,
                "mask_token": self.mask_token,
            }
        )
        return config
