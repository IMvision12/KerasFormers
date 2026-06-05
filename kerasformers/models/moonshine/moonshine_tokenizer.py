import os
from typing import List, Union

import keras
import numpy as np
from tokenizers import Tokenizer

from kerasformers.base import BaseTokenizer

from .config import MOONSHINE_HF_REPO


def _resolve_tokenizer_file(tokenizer_file, hf_id):
    if tokenizer_file is not None and os.path.exists(tokenizer_file):
        return tokenizer_file
    from huggingface_hub import hf_hub_download

    return hf_hub_download(hf_id, "tokenizer.json")


@keras.saving.register_keras_serializable(package="kerasformers")
class MoonshineTokenizer(BaseTokenizer):
    """Moonshine SentencePiece-BPE tokenizer, built on the ``tokenizers`` library.

    Loads the canonical ``tokenizer.json`` shipped with the Useful Sensors
    checkpoints — a byte-fallback BPE with a metaspace (``▁``) normalizer and a
    template post-processor that prepends ``<s>``. The file is downloaded from
    the Hub repo (``UsefulSensors/moonshine-*``) unless an explicit
    ``tokenizer_file`` path is given.

    The encode path (used for label preparation) does **not** add special
    tokens; ``MoonshineSpeechToText`` seeds decoding with
    ``decoder_start_token_id`` itself. Decoding drops the ``<s>`` / ``</s>`` /
    ``<unk>`` specials by default.

    Args:
        tokenizer_file: Optional explicit path to ``tokenizer.json``. When
            ``None``, the file is downloaded from ``hf_id``.
        hf_id: Hub repo to download ``tokenizer.json`` from when
            ``tokenizer_file`` is not supplied.
        bos_token_id / eos_token_id / unk_token_id: Moonshine special ids.
    """

    def __init__(
        self,
        tokenizer_file: str = None,
        hf_id: str = MOONSHINE_HF_REPO["moonshine_tiny"],
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        tokenizer_file = _resolve_tokenizer_file(tokenizer_file, hf_id)
        self.tokenizer_file = tokenizer_file
        self.hf_id = hf_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

        self._tok = Tokenizer.from_file(tokenizer_file)
        self._special_id_set = {bos_token_id, eos_token_id, unk_token_id}

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self._tok.encode(text, add_special_tokens=False).ids
        encs = self._tok.encode_batch(text, add_special_tokens=False)
        return [e.ids for e in encs]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids = [int(i) for i in token_ids]
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._special_id_set]
        return self._tok.decode(ids, skip_special_tokens=False)

    def batch_decode(
        self, token_ids_batch, skip_special_tokens: bool = True
    ) -> List[str]:
        if hasattr(token_ids_batch, "numpy"):
            token_ids_batch = token_ids_batch.numpy()
        out = []
        for row in token_ids_batch:
            row = row.tolist() if hasattr(row, "tolist") else list(row)
            out.append(self.decode(row, skip_special_tokens=skip_special_tokens))
        return out

    def call(self, inputs: Union[str, List[str]]):
        if inputs is None:
            raise ValueError("No text inputs provided to MoonshineTokenizer")
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        encs = self._tok.encode_batch(texts, add_special_tokens=False)
        lens = [len(e.ids) for e in encs]
        max_len = max(lens) if lens else 0
        ids = np.full((len(texts), max_len), self.eos_token_id, dtype=np.int32)
        mask = np.zeros((len(texts), max_len), dtype=np.int32)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = 1
        return {
            "input_ids": keras.ops.convert_to_tensor(ids, dtype="int32"),
            "attention_mask": keras.ops.convert_to_tensor(mask, dtype="int32"),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer_file": self.tokenizer_file,
                "hf_id": self.hf_id,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "unk_token_id": self.unk_token_id,
            }
        )
        return config
