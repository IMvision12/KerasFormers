from typing import List, Union

import keras
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import Unigram
from tokenizers.normalizers import Precompiled
from tokenizers.pre_tokenizers import Metaspace, WhitespaceSplit
from tokenizers.pre_tokenizers import Sequence as PreSeq
from tokenizers.processors import TemplateProcessing

from kerasformers.base import BaseTokenizer
from kerasformers.conversion import download_file

from .config import XLM_ROBERTA_VOCAB_URL


@keras.saving.register_keras_serializable(package="kerasformers")
class XLMRobertaTokenizer(BaseTokenizer):
    """XLM-RoBERTa SentencePiece tokenizer, built on the `tokenizers` library.

    Reads XLM-RoBERTa's ``sentencepiece.bpe.model`` and assembles a
    `tokenizers.Tokenizer` that matches Hugging Face exactly: the SentencePiece
    pieces are remapped with the fairseq id offset (``<s>``=0, ``<pad>``=1,
    ``</s>``=2, ``<unk>``=3, then each SentencePiece id shifted by +1, and
    ``<mask>`` last), the model's own ``Precompiled`` normalizer + ``▁``
    metaspace pre-tokenizer are reused, and ``<s> A </s>`` /
    ``<s> A </s> </s> B </s>`` post-processing is applied. ``call`` returns the
    ``input_ids`` / ``attention_mask`` / ``token_type_ids`` dict expected by
    :class:`XLMRobertaModel` (token types are always ``0`` for XLM-RoBERTa).

    Args:
        vocab_file: Path to ``sentencepiece.bpe.model``. When ``None``, downloads
            the default kerasformers-release model on first use.
        max_seq_len: Truncation length (default 512). Batches are padded to the
            longest sequence, so short inputs stay short.
        bos_token / eos_token / unk_token / pad_token / mask_token: Special tokens.
    """

    def __init__(
        self,
        vocab_file: str = None,
        max_seq_len: int = 512,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vocab_file is None:
            vocab_file = download_file(XLM_ROBERTA_VOCAB_URL)
        self.vocab_file = vocab_file
        self.max_seq_len = max_seq_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        sp = spm.SentencePieceProcessor()
        sp.load(vocab_file)
        vocab = [
            (bos_token, 0.0),
            (pad_token, 0.0),
            (eos_token, 0.0),
            (unk_token, 0.0),
        ]
        for piece_id in range(3, sp.get_piece_size()):
            vocab.append((sp.id_to_piece(piece_id), sp.get_score(piece_id)))
        vocab.append((mask_token, 0.0))

        proto = sp_pb2.ModelProto()
        proto.ParseFromString(open(vocab_file, "rb").read())

        tok = Tokenizer(
            Unigram(
                vocab,
                unk_id=3,
                byte_fallback=proto.trainer_spec.byte_fallback,
            )
        )
        tok.normalizer = Precompiled(proto.normalizer_spec.precompiled_charsmap)
        tok.pre_tokenizer = PreSeq(
            [WhitespaceSplit(), Metaspace(replacement="▁", prepend_scheme="always")]
        )
        tok.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            pair=f"{bos_token} $A {eos_token} {eos_token} $B {eos_token}",
            special_tokens=[(bos_token, 0), (eos_token, 2)],
        )
        tok.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")
        tok.add_special_tokens(
            [
                AddedToken(bos_token, special=True, normalized=False),
                AddedToken(pad_token, special=True, normalized=False),
                AddedToken(eos_token, special=True, normalized=False),
                AddedToken(unk_token, special=True, normalized=False),
                AddedToken(mask_token, special=True, normalized=False, lstrip=True),
            ]
        )
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(pad_id=tok.token_to_id(pad_token), pad_token=pad_token)
        self._tok = tok

        self.bos_token_id = tok.token_to_id(bos_token)
        self.eos_token_id = tok.token_to_id(eos_token)
        self.unk_token_id = tok.token_to_id(unk_token)
        self.pad_token_id = tok.token_to_id(pad_token)
        self.mask_token_id = tok.token_to_id(mask_token)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        """Load an XLM-RoBERTa finetune's ``sentencepiece.bpe.model`` from the HF
        ``repo`` instead of the bundled kerasformers-release default."""
        from huggingface_hub import hf_hub_download

        return cls(
            vocab_file=hf_hub_download(repo, "sentencepiece.bpe.model"), **kwargs
        )

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
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "mask_token": self.mask_token,
            }
        )
        return config
