import json
from typing import Dict, List, Optional, Union

import keras
import numpy as np
import sentencepiece as spm
from keras import ops

from kerasformers.base import BaseTokenizer
from kerasformers.weight_utils import download_file

from .config import SPEECH2TEXT_TOKENIZER_FILES

SPIECE_UNDERLINE = "▁"


def _resolve_tokenizer_files(vocab_file, spm_file):
    if vocab_file is None and spm_file is None:
        vocab_file = download_file(SPEECH2TEXT_TOKENIZER_FILES["vocab"])
        spm_file = download_file(SPEECH2TEXT_TOKENIZER_FILES["spm"])
    return vocab_file, spm_file


@keras.saving.register_keras_serializable(package="kerasformers")
class Speech2TextTokenizer(BaseTokenizer):
    """Speech2Text SentencePiece tokenizer.

    Reproduces the reference ``Speech2TextTokenizer``: a SentencePiece model
    turns text into subword pieces, and a separate ``vocab.json`` maps those
    pieces to ids. Mainly used to **decode** generated ids back to text
    (``ids -> pieces -> SentencePiece decode``); the encode path is provided
    for label preparation. The LibriSpeech vocabulary is lowercase and uses
    ``</s>`` (id 2) as both the decoder start token and the end-of-sequence
    token (Bart convention).

    Args:
        vocab_file: Path to ``vocab.json`` (token -> id). Downloaded from the
            HF repo when ``None``.
        spm_file: Path to the SentencePiece ``.model`` file. Downloaded when
            ``None``.
        do_upper_case: Upper-case the decoded text (multilingual ST variants).
        do_lower_case: Lower-case the input text before encoding.
        max_seq_len: Maximum target length (used when padding label ids).
        bos_token / eos_token / pad_token / unk_token: Special token strings.
    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        spm_file: Optional[str] = None,
        do_upper_case: bool = False,
        do_lower_case: bool = False,
        max_seq_len: int = 1024,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        vocab_file, spm_file = _resolve_tokenizer_files(vocab_file, spm_file)
        self.vocab_file = vocab_file
        self.spm_file = spm_file
        self.do_upper_case = do_upper_case
        self.do_lower_case = do_lower_case
        self.max_seq_len = max_seq_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        with open(vocab_file, encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)
        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(spm_file)

        self.bos_token_id = self.encoder[bos_token]
        self.eos_token_id = self.encoder[eos_token]
        self.pad_token_id = self.encoder[pad_token]
        self.unk_token_id = self.encoder[unk_token]
        self._special_ids = {
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.unk_token_id,
        }

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def tokenize(self, text: Union[str, List[str]]):
        def _one(t):
            if self.do_lower_case:
                t = t.lower()
            pieces = self.sp_model.encode(t, out_type=str)
            return [self.encoder.get(p, self.unk_token_id) for p in pieces]

        if isinstance(text, str):
            return _one(text)
        return [_one(t) for t in text]

    def detokenize(self, token_ids, skip_special_tokens: bool = True) -> str:
        if hasattr(token_ids, "numpy"):
            token_ids = token_ids.numpy()
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        pieces: List[str] = []
        for i in token_ids:
            i = int(i)
            if skip_special_tokens and i in self._special_ids:
                continue
            pieces.append(self.decoder.get(i, self.unk_token))
        text = self.sp_model.decode(pieces) if pieces else ""
        if self.do_upper_case:
            text = text.upper()
        return text.strip()

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.detokenize(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self, token_ids_batch, skip_special_tokens: bool = True
    ) -> List[str]:
        if hasattr(token_ids_batch, "numpy"):
            token_ids_batch = token_ids_batch.numpy()
        out = []
        for row in token_ids_batch:
            row = row.tolist() if hasattr(row, "tolist") else list(row)
            out.append(self.detokenize(row, skip_special_tokens=skip_special_tokens))
        return out

    batch_detokenize = batch_decode

    def prepare_for_model(self, text: Union[str, List[int]]) -> Dict[str, List[int]]:
        ids = self.tokenize(text) if isinstance(text, str) else list(text)
        ids = ids[: self.max_seq_len - 1] + [self.eos_token_id]
        return {"input_ids": ids}

    def call(self, inputs):
        if inputs is None:
            raise ValueError("No text inputs provided to Speech2TextTokenizer")
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        seqs = [self.tokenize(t) + [self.eos_token_id] for t in texts]
        max_len = max(len(s) for s in seqs)
        padded = [s + [self.pad_token_id] * (max_len - len(s)) for s in seqs]
        ids = np.array(padded, dtype=np.int32)
        return {"input_ids": ops.convert_to_tensor(ids, dtype="int32")}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_file": self.vocab_file,
                "spm_file": self.spm_file,
                "do_upper_case": self.do_upper_case,
                "do_lower_case": self.do_lower_case,
                "max_seq_len": self.max_seq_len,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
            }
        )
        return config
