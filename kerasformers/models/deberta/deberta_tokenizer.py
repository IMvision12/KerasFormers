import json
from typing import List, Union

import keras
import numpy as np
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing

from kerasformers.base import BaseTokenizer
from kerasformers.weight_utils import download_file

from .config import DEBERTA_MERGES_URL, DEBERTA_VOCAB_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class DebertaTokenizer(BaseTokenizer):
    """DeBERTa (v1) GPT-2 byte-level BPE tokenizer, built on the `tokenizers` lib.

    DeBERTa v1 uses GPT-2's byte-level BPE over ``vocab.json`` + ``merges.txt``
    but BERT-style special tokens and ``[CLS] A [SEP]`` /
    ``[CLS] A [SEP] B [SEP]`` post-processing. ``call`` returns the ``input_ids``
    / ``attention_mask`` / ``token_type_ids`` dict expected by
    :class:`DebertaModel` (``token_type_ids`` is produced for parity but the
    model has no token-type embeddings).

    Args:
        vocab_file: Path to ``vocab.json``. When ``None`` (and ``merges_file`` is
            also ``None``), downloads the default kerasformers-release files.
        merges_file: Path to ``merges.txt``. See ``vocab_file``.
        max_seq_len: Truncation length (default 512). Batches are padded to the
            longest sequence.
        cls_token / sep_token / pad_token / unk_token / mask_token: Special tokens.
    """

    def __init__(
        self,
        vocab_file: str = None,
        merges_file: str = None,
        max_seq_len: int = 512,
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vocab_file is None and merges_file is None:
            vocab_file = download_file(DEBERTA_VOCAB_URL)
            merges_file = download_file(DEBERTA_MERGES_URL)
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.max_seq_len = max_seq_len
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.cls_token_id = self.encoder[cls_token]
        self.sep_token_id = self.encoder[sep_token]
        self.pad_token_id = self.encoder[pad_token]
        self.unk_token_id = self.encoder[unk_token]
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
        tok.post_processor = TemplateProcessing(
            single=f"{cls_token}:0 $A:0 {sep_token}:0",
            pair=f"{cls_token}:0 $A:0 {sep_token}:0 {sep_token}:0 $B:1 {sep_token}:1",
            special_tokens=[
                (cls_token, self.cls_token_id),
                (sep_token, self.sep_token_id),
            ],
        )
        tok.decoder = ByteLevelDecoder()
        tok.add_special_tokens(
            [
                AddedToken(t, special=True, normalized=False)
                for t in (cls_token, sep_token, pad_token, unk_token, mask_token)
            ]
        )
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(pad_id=self.pad_token_id, pad_token=pad_token)
        self._tok = tok

    @classmethod
    def from_hf(cls, repo, **kwargs):
        """Load a DeBERTa finetune's ``vocab.json`` + ``merges.txt`` from the HF
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
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self._tok.decode(
            [int(i) for i in ids], skip_special_tokens=skip_special_tokens
        )

    def call(self, inputs: Union[str, List[str]], text_pair=None):
        if inputs is None:
            raise ValueError("No text inputs provided to DebertaTokenizer")
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        if text_pair is None:
            encs = self._tok.encode_batch(texts)
        else:
            pairs = [text_pair] if isinstance(text_pair, str) else list(text_pair)
            encs = self._tok.encode_batch(list(zip(texts, pairs)))

        input_ids = np.array([e.ids for e in encs], dtype=np.int32)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int32)
        token_type_ids = np.array([e.type_ids for e in encs], dtype=np.int32)
        return {
            "input_ids": keras.ops.convert_to_tensor(input_ids, dtype="int32"),
            "attention_mask": keras.ops.convert_to_tensor(
                attention_mask, dtype="int32"
            ),
            "token_type_ids": keras.ops.convert_to_tensor(
                token_type_ids, dtype="int32"
            ),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_file": self.vocab_file,
                "merges_file": self.merges_file,
                "max_seq_len": self.max_seq_len,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "mask_token": self.mask_token,
            }
        )
        return config
