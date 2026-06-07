import os
from typing import List, Union

import keras
import numpy as np

from kerasformers.base import BaseTokenizer
from kerasformers.conversion import download_file

METACLIP2_EOS_TOKEN_ID = 2
METACLIP2_BOS_TOKEN_ID = 0
METACLIP2_PAD_TOKEN_ID = 1
METACLIP2_UNK_TOKEN_ID = 3
METACLIP2_MASK_TOKEN_ID = 901628

DEFAULT_SENTENCEPIECE_URL = (
    "https://github.com/IMvision12/KerasFormers/releases/download/metaclip2/"
    "sentencepiece.bpe.model"
)


@keras.saving.register_keras_serializable(package="kerasformers")
class MetaClip2Tokenizer(BaseTokenizer):
    """XLM-RoBERTa SentencePiece tokenizer for MetaCLIP 2 worldwide variants.

    Wraps :class:`sentencepiece.SentencePieceProcessor` to match the reference's
    ``XLMRobertaTokenizer`` bit-close on the multilingual MetaCLIP 2
    checkpoints (901 629-token vocab covering ~300 languages).

    Tokenization pipeline:

    1. SentencePiece encode the raw text to integer pieces.
    2. **Apply the fairseq offset of +1** to every piece id — XLM-R
       reserves token ids ``0..3`` for ``<s> / <pad> / </s> / <unk>``
       and shifts the SP vocabulary by one to make room. Without this
       offset the model would see the wrong embeddings.
    3. Prepend ``bos_token_id`` and append ``eos_token_id``.
    4. Truncate (replacing the last token with EOS) or pad with
       ``pad_token_id`` to ``max_seq_len``.
    5. Build an attention mask: ``1`` for real tokens, ``0`` for
       padding.

    Special-token ids match XLM-R exactly: BOS=0, PAD=1, EOS=2, UNK=3,
    MASK=901628. The MASK id is larger than EOS, which is why
    :func:`metaclip2_text_backbone` pools by explicit ``token == EOS``
    match rather than ``argmax``.

    Args:
        sentencepiece_model_file: Path to ``sentencepiece.bpe.model``.
            When ``None``, downloads it from the default MetaCLIP 2
            release URL on first use.
        max_seq_len: Maximum sequence length. Defaults to ``77``
            (the value used by every MetaCLIP 2 checkpoint).
        bos_token_id: BOS token id. Defaults to ``0``.
        eos_token_id: EOS token id. Defaults to ``2``.
        pad_token_id: PAD token id. Defaults to ``1``.
        unk_token_id: UNK token id. Defaults to ``3``.

    Example:
        >>> from kerasformers.models.metaclip2 import MetaClip2Tokenizer
        >>> tok = MetaClip2Tokenizer.from_weights("metaclip2_worldwide_s16_224")
        >>> out = tok(["un chat", "a cat"])
        >>> out["token_ids"].shape       # (2, 77)
        >>> out["padding_mask"].shape    # (2, 77) — 1 for real tokens
    """

    def __init__(
        self,
        sentencepiece_model_file: str = None,
        max_seq_len: int = 77,
        bos_token_id: int = METACLIP2_BOS_TOKEN_ID,
        eos_token_id: int = METACLIP2_EOS_TOKEN_ID,
        pad_token_id: int = METACLIP2_PAD_TOKEN_ID,
        unk_token_id: int = METACLIP2_UNK_TOKEN_ID,
        **kwargs,
    ):
        super().__init__(**kwargs)

        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "MetaClip2Tokenizer requires the `sentencepiece` package. "
                "Install with `pip install sentencepiece`."
            ) from exc

        if sentencepiece_model_file is None:
            sentencepiece_model_file = download_file(DEFAULT_SENTENCEPIECE_URL)
        if not os.path.exists(sentencepiece_model_file):
            raise FileNotFoundError(sentencepiece_model_file)

        self.sentencepiece_model_file = sentencepiece_model_file
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(sentencepiece_model_file)

        self.fairseq_offset = 1

    def _encode_one(self, text: str) -> List[int]:
        pieces = self._sp.encode(text, out_type=int)
        ids = [p + self.fairseq_offset for p in pieces]
        ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len - 1] + [self.eos_token_id]
        pad_len = self.max_seq_len - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.pad_token_id] * pad_len
        return ids, attention_mask

    def call(self, inputs: Union[str, List[str]]):
        texts = self.normalize_texts(inputs)
        ids = np.zeros((len(texts), self.max_seq_len), dtype=np.int32)
        mask = np.zeros((len(texts), self.max_seq_len), dtype=np.int32)
        for i, t in enumerate(texts):
            row_ids, row_mask = self._encode_one(t)
            ids[i] = row_ids
            mask[i] = row_mask
        return {
            "token_ids": keras.ops.convert_to_tensor(ids, dtype="int32"),
            "padding_mask": keras.ops.convert_to_tensor(mask, dtype="int32"),
        }

    def decode(self, ids) -> List[str]:
        if hasattr(ids, "numpy"):
            ids = ids.numpy()
        ids = np.asarray(ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        out = []
        for row in ids:
            keep = [
                int(i) - self.fairseq_offset
                for i in row
                if int(i)
                not in (
                    self.bos_token_id,
                    self.eos_token_id,
                    self.pad_token_id,
                )
                and int(i) >= self.fairseq_offset
            ]
            out.append(self._sp.decode(keep))
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sentencepiece_model_file": self.sentencepiece_model_file,
                "max_seq_len": self.max_seq_len,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
                "unk_token_id": self.unk_token_id,
            }
        )
        return config
