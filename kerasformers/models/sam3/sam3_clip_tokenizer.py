import json

import numpy as np
from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, Lowercase, Replace
from tokenizers.normalizers import Sequence as NormSeq
from tokenizers.pre_tokenizers import ByteLevel, Split
from tokenizers.pre_tokenizers import Sequence as PreSeq
from tokenizers.processors import RobertaProcessing

from kerasformers.base import BaseTokenizer
from kerasformers.weight_utils import download_file

VOCAB_URL = (
    "https://github.com/IMvision12/KerasFormers/releases/download/clip/vocab.json"
)
MERGES_URL = (
    "https://github.com/IMvision12/KerasFormers/releases/download/clip/merges.txt"
)

SAM3_CONTEXT_LENGTH = 32
SAM3_VOCAB_SIZE = 49408
SAM3_BOS_TOKEN_ID = 49406
SAM3_EOS_TOKEN_ID = 49407
SAM3_PAD_TOKEN_ID = 49407


class SAM3CLIPTokenizer(BaseTokenizer):
    """BPE tokenizer for SAM3's CLIP text encoder (max_seq_len=32).

    Uses the `tokenizers` library (Rust) under the hood. Auto-downloads
    ``vocab.json`` and ``merges.txt`` on first use.

    Args:
        vocab_file: Path to vocab.json. If None, auto-downloads.
        merges_file: Path to merges.txt. If None, auto-downloads.
        max_seq_len: Max sequence length (default 32 for SAM3).

    Usage:
        tokenizer = SAM3CLIPTokenizer()
        input_ids, attention_mask = tokenizer.encode("a cat")
        # input_ids: (1, 32) int32, attention_mask: (1, 32) float32
    """

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        max_seq_len=SAM3_CONTEXT_LENGTH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.bos_token_id = SAM3_BOS_TOKEN_ID
        self.eos_token_id = SAM3_EOS_TOKEN_ID
        self.pad_token_id = SAM3_PAD_TOKEN_ID

        if vocab_file is None:
            vocab_file = download_file(VOCAB_URL)
        if merges_file is None:
            merges_file = download_file(MERGES_URL)

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        tok = Tokenizer(
            BPE(
                vocab=vocab_file,
                merges=merges_file,
                unk_token="<|endoftext|>",
                end_of_word_suffix="</w>",
                fuse_unk=False,
            )
        )
        tok.normalizer = NormSeq([NFC(), Replace(Regex(r"\s+"), " "), Lowercase()])
        tok.pre_tokenizer = PreSeq(
            [
                Split(
                    pattern=Regex(
                        r"<\|startoftext\|>|<\|endoftext\|>"
                        r"|'s|'t|'re|'ve|'m|'ll|'d"
                        r"|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
                    ),
                    behavior="removed",
                    invert=True,
                ),
                ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
            ]
        )
        tok.post_processor = RobertaProcessing(
            sep=("<|endoftext|>", self.eos_token_id),
            cls=("<|startoftext|>", self.bos_token_id),
            trim_offsets=False,
            add_prefix_space=False,
        )
        tok.decoder = ByteLevelDecoder()
        tok.add_special_tokens(
            [
                AddedToken("<|startoftext|>", special=True, normalized=False),
                AddedToken("<|endoftext|>", special=True, normalized=False),
            ]
        )
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(
            pad_id=self.pad_token_id,
            pad_token="<|endoftext|>",
            length=max_seq_len,
        )
        self._tok = tok

    def encode(self, text):
        """Tokenize text and return padded input_ids + attention_mask.

        Args:
            text: str or list of str.

        Returns:
            input_ids: numpy array (batch, max_seq_len) int32.
            attention_mask: numpy array (batch, max_seq_len) float32.
        """
        if isinstance(text, str):
            text = [text]
        encs = self._tok.encode_batch(list(text))
        input_ids = np.array([e.ids for e in encs], dtype=np.int32)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.float32)
        return input_ids, attention_mask

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        skip = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        keep = [int(i) for i in token_ids if int(i) not in skip]
        text = self._tok.decode(keep, skip_special_tokens=False)
        return text.replace("</w>", " ").strip()
