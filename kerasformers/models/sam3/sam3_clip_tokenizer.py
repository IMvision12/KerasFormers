import keras
import numpy as np
from tokenizers import Tokenizer

from kerasformers.base import BaseTokenizer

from .config import SAM3_TOKENIZER_URLS

SAM3_CONTEXT_LENGTH = 32
SAM3_VOCAB_SIZE = 49408
SAM3_BOS_TOKEN_ID = 49406
SAM3_EOS_TOKEN_ID = 49407
SAM3_PAD_TOKEN_ID = 49407


@keras.saving.register_keras_serializable(package="kerasformers")
class SAM3CLIPTokenizer(BaseTokenizer):
    """BPE tokenizer for SAM3's CLIP text encoder (max_seq_len=32).

    Loads the OpenAI CLIP fast-tokenizer ``tokenizer.json`` for ``variant`` from the
    ``clip`` release tag (or an explicit ``tokenizer_file``) and re-enables CLIP's
    truncation + ``<|endoftext|>`` padding to ``max_seq_len`` (32 for SAM3). SAM3
    reuses the OpenAI CLIP tokenizer.

    Args:
        variant: SAM3 variant key (default ``"sam3_saco"``).
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides variant).
        max_seq_len: Max sequence length (default 32 for SAM3).

    Usage:
        tokenizer = SAM3CLIPTokenizer.from_weights("sam3_saco")
        input_ids, attention_mask = tokenizer.encode("a cat")
        # input_ids: (1, 32) int32, attention_mask: (1, 32) float32
    """

    TOKENIZER_URLS = SAM3_TOKENIZER_URLS
    DEFAULT_VARIANT = "sam3_saco"

    def __init__(
        self,
        variant=None,
        tokenizer_file=None,
        max_seq_len=SAM3_CONTEXT_LENGTH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant or self.DEFAULT_VARIANT
        tokenizer_file = self.resolve_tokenizer_json(self.variant, tokenizer_file)
        self.tokenizer_file = tokenizer_file
        self.max_seq_len = max_seq_len
        self.bos_token_id = SAM3_BOS_TOKEN_ID
        self.eos_token_id = SAM3_EOS_TOKEN_ID
        self.pad_token_id = SAM3_PAD_TOKEN_ID

        tok = Tokenizer.from_file(tokenizer_file)
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(
            pad_id=self.pad_token_id, pad_token="<|endoftext|>", length=max_seq_len
        )
        self._tok = tok

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        return cls(tokenizer_file=hf_hub_download(repo, "tokenizer.json"), **kwargs)

    def encode(self, text):
        texts = self.normalize_texts(text)
        encs = self._tok.encode_batch(texts)
        input_ids = np.array([e.ids for e in encs], dtype=np.int32)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.float32)
        return input_ids, attention_mask

    def decode(self, token_ids):
        skip = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        keep = [i for i in self.to_id_list(token_ids) if i not in skip]
        text = self._tok.decode(keep, skip_special_tokens=False)
        return text.replace("</w>", " ").strip()
