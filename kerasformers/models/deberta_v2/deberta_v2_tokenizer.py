from typing import List, Union

import keras
import numpy as np
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2
from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import Unigram
from tokenizers.normalizers import Precompiled, Replace, Strip
from tokenizers.normalizers import Sequence as NormSeq
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing

from kerasformers.base import BaseTokenizer
from kerasformers.conversion import download_file

from .config import DEBERTA_V2_VOCAB_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class DebertaV2Tokenizer(BaseTokenizer):
    """DeBERTa-v2 / v3 SentencePiece tokenizer, built on the `tokenizers` library.

    Reads DeBERTa-v2's ``spm.model`` and assembles a `tokenizers.Tokenizer` that
    matches Hugging Face: SentencePiece piece ids map directly to vocab ids (no
    fairseq offset), ``[MASK]`` is appended after the SentencePiece vocabulary,
    the model's own ``Precompiled`` normalizer + ``▁`` metaspace pre-tokenizer
    are reused, and ``[CLS] A [SEP]`` / ``[CLS] A [SEP] B [SEP]`` post-processing
    is applied. ``call`` returns the ``input_ids`` / ``attention_mask`` /
    ``token_type_ids`` dict expected by :class:`DebertaV2Model`.

    Args:
        vocab_file: Path to ``spm.model``. When ``None``, downloads the default
            kerasformers-release model on first use.
        max_seq_len: Truncation length (default 512).
        cls_token / sep_token / pad_token / unk_token / mask_token: Special tokens.
    """

    def __init__(
        self,
        vocab_file: str = None,
        max_seq_len: int = 512,
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vocab_file is None:
            vocab_file = download_file(DEBERTA_V2_VOCAB_URL)
        self.vocab_file = vocab_file
        self.max_seq_len = max_seq_len
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token

        sp = spm.SentencePieceProcessor()
        sp.load(vocab_file)
        vocab = [
            (sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())
        ]
        vocab.append((mask_token, 0.0))

        proto = sp_pb2.ModelProto()
        proto.ParseFromString(open(vocab_file, "rb").read())

        tok = Tokenizer(
            Unigram(
                vocab,
                unk_id=sp.piece_to_id(unk_token),
                byte_fallback=proto.trainer_spec.byte_fallback,
            )
        )
        tok.normalizer = NormSeq(
            [
                Precompiled(proto.normalizer_spec.precompiled_charsmap),
                Replace(Regex(r"\s+"), " "),
                Strip(),
            ]
        )
        tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
        tok.post_processor = TemplateProcessing(
            single=f"{cls_token}:0 $A:0 {sep_token}:0",
            pair=f"{cls_token}:0 $A:0 {sep_token}:0 $B:1 {sep_token}:1",
            special_tokens=[
                (cls_token, sp.piece_to_id(cls_token)),
                (sep_token, sp.piece_to_id(sep_token)),
            ],
        )
        tok.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")
        tok.add_special_tokens(
            [
                AddedToken(t, special=True, normalized=False)
                for t in (cls_token, sep_token, pad_token, unk_token, mask_token)
            ]
        )
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

        return cls(vocab_file=hf_hub_download(repo, "spm.model"), **kwargs)

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
            raise ValueError("No text inputs provided to DebertaV2Tokenizer")
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
                "max_seq_len": self.max_seq_len,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "mask_token": self.mask_token,
            }
        )
        return config
