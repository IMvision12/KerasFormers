from typing import List, Union

import keras
import numpy as np
from tokenizers import Tokenizer

from kerasformers.base import BaseTokenizer
from kerasformers.weight_utils import download_file

from .config import MODERNBERT_VOCAB_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class ModernBertTokenizer(BaseTokenizer):
    """ModernBERT byte-level BPE tokenizer, built on the `tokenizers` library.

    Loads ModernBERT's ``tokenizer.json`` (a complete fast-tokenizer spec, with
    its byte-level BPE model, ``[CLS] A [SEP]`` / ``[CLS] A [SEP] B [SEP]``
    post-processing and special tokens already baked in) and adds runtime
    truncation + padding. ``call`` returns the ``input_ids`` / ``attention_mask``
    dict expected by :class:`ModernBertModel` — ModernBERT has no token-type
    embeddings, so no ``token_type_ids`` are produced.

    Args:
        vocab_file: Path to ``tokenizer.json``. When ``None``, downloads the
            default kerasformers-release tokenizer on first use.
        max_seq_len: Truncation length (default 8192).
        cls_token / sep_token / pad_token / unk_token / mask_token: Special tokens.
    """

    def __init__(
        self,
        vocab_file: str = None,
        max_seq_len: int = 8192,
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vocab_file is None:
            vocab_file = download_file(MODERNBERT_VOCAB_URL)
        self.vocab_file = vocab_file
        self.max_seq_len = max_seq_len
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token

        tok = Tokenizer.from_file(vocab_file)
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(pad_id=tok.token_to_id(pad_token), pad_token=pad_token)
        self._tok = tok

        self.cls_token_id = tok.token_to_id(cls_token)
        self.sep_token_id = tok.token_to_id(sep_token)
        self.pad_token_id = tok.token_to_id(pad_token)
        self.unk_token_id = tok.token_to_id(unk_token)
        self.mask_token_id = tok.token_to_id(mask_token)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        return cls(vocab_file=hf_hub_download(repo, "tokenizer.json"), **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

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
            raise ValueError("No text inputs provided to ModernBertTokenizer")
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        if text_pair is None:
            encs = self._tok.encode_batch(texts)
        else:
            pairs = [text_pair] if isinstance(text_pair, str) else list(text_pair)
            encs = self._tok.encode_batch(list(zip(texts, pairs)))

        input_ids = np.array([e.ids for e in encs], dtype=np.int32)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int32)
        return {
            "input_ids": keras.ops.convert_to_tensor(input_ids, dtype="int32"),
            "attention_mask": keras.ops.convert_to_tensor(
                attention_mask, dtype="int32"
            ),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_file": self.vocab_file,
                "max_seq_len": self.max_seq_len,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "mask_token": self.mask_token,
            }
        )
        return config
