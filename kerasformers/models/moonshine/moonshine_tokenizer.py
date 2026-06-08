import os
from typing import List, Union

import keras
from tokenizers import Tokenizer

from kerasformers.base import BaseTokenizer
from kerasformers.conversion import download_file

from .config import MOONSHINE_TOKENIZER_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class MoonshineTokenizer(BaseTokenizer):
    """Moonshine SentencePiece-BPE tokenizer, built on the ``tokenizers`` library.

    Loads the canonical ``tokenizer.json`` shipped with the Useful Sensors
    checkpoints — a byte-fallback BPE with a metaspace (``▁``) normalizer and a
    template post-processor that prepends ``<s>``. The file is pulled from the
    ``moonshine`` release tag on ``github.com/IMvision12/KerasFormers`` unless an
    explicit ``tokenizer_file`` path is given (tiny and base share one vocab).

    The encode path (used for label preparation) does **not** add special
    tokens; ``MoonshineSpeechToText`` seeds decoding with
    ``decoder_start_token_id`` itself. Decoding drops the ``<s>`` / ``</s>`` /
    ``<unk>`` specials by default.

    Args:
        tokenizer_file: Optional explicit path to ``tokenizer.json``. When
            ``None``, the default kerasformers-release file is downloaded.
        bos_token_id / eos_token_id / unk_token_id: Moonshine special ids.
    """

    def __init__(
        self,
        tokenizer_file: str = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if tokenizer_file is None or not os.path.exists(tokenizer_file):
            tokenizer_file = download_file(MOONSHINE_TOKENIZER_URL)
        self.tokenizer_file = tokenizer_file
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

        self._tok = Tokenizer.from_file(tokenizer_file)
        self._special_id_set = {bos_token_id, eos_token_id, unk_token_id}

    @classmethod
    def from_hf(cls, repo, **kwargs):
        """Load a finetune's ``tokenizer.json`` from the HF ``repo`` instead of the
        bundled kerasformers-release default."""
        from huggingface_hub import hf_hub_download

        return cls(tokenizer_file=hf_hub_download(repo, "tokenizer.json"), **kwargs)

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
        ids = self.to_id_list(token_ids)
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
        texts = self.normalize_texts(inputs)
        encs = self._tok.encode_batch(texts, add_special_tokens=False)
        ids, mask = self.pad_batch([e.ids for e in encs], pad_value=self.eos_token_id)
        return {
            "input_ids": keras.ops.convert_to_tensor(ids, dtype="int32"),
            "attention_mask": keras.ops.convert_to_tensor(mask, dtype="int32"),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer_file": self.tokenizer_file,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "unk_token_id": self.unk_token_id,
            }
        )
        return config
