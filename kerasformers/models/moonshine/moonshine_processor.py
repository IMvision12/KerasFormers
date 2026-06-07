from typing import List, Optional, Union

import keras

from kerasformers.base import BaseProcessor

from .config import MOONSHINE_HF_REPO
from .moonshine_feature_extractor import MoonshineFeatureExtractor
from .moonshine_tokenizer import MoonshineTokenizer


@keras.saving.register_keras_serializable(package="kerasformers")
class MoonshineProcessor(BaseProcessor):
    """Combined audio + text processor for Moonshine.

    Wraps :class:`MoonshineFeatureExtractor` and :class:`MoonshineTokenizer`,
    matching the reference ``MoonshineProcessor`` (a ``Wav2Vec2FeatureExtractor``
    + the Moonshine tokenizer):

    * ``processor(audio=..., sampling_rate=16000)`` -> raw-waveform
      ``input_values`` (zero-padded batch).
    * ``processor(text=...)`` -> token ids + attention mask (label path).
    * ``processor.decode`` / ``processor.batch_decode`` -> proxy to the
      tokenizer.

    ``decoder_start_token_id`` is ``<s>`` (id 1) — the token
    :class:`~kerasformers.models.moonshine.MoonshineSpeechToText` seeds greedy
    decoding with.

    Args:
        tokenizer_file: Optional explicit ``tokenizer.json`` path. Downloaded
            from ``hf_id`` when ``None``.
        hf_id: Hub repo to fetch ``tokenizer.json`` from.
        sampling_rate: Forwarded to the feature extractor.
        decoder_start_token_id: Seed token id for generation (``<s>`` = 1).
        bos_token_id / eos_token_id / unk_token_id: Forwarded to the tokenizer.
    """

    def __init__(
        self,
        tokenizer_file: Optional[str] = None,
        hf_id: str = MOONSHINE_HF_REPO["moonshine_tiny"],
        sampling_rate: int = 16000,
        decoder_start_token_id: int = 1,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_extractor = MoonshineFeatureExtractor(sampling_rate=sampling_rate)
        self.tokenizer = MoonshineTokenizer(
            tokenizer_file=tokenizer_file,
            hf_id=hf_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
        )
        self.decoder_start_token_id = decoder_start_token_id

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self, token_ids_batch, skip_special_tokens: bool = True
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            token_ids_batch, skip_special_tokens=skip_special_tokens
        )

    def call(
        self,
        audio=None,
        text: Union[str, List[str], None] = None,
        sampling_rate: int = 16000,
    ):
        if audio is None and text is None:
            raise ValueError(
                "At least one of 'audio' or 'text' must be provided to "
                "MoonshineProcessor"
            )
        out = {}
        if audio is not None:
            out["input_values"] = self.feature_extractor(
                audio, sampling_rate=sampling_rate
            )
        if text is not None:
            out.update(self.tokenizer(text))
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer_file": self.tokenizer.tokenizer_file,
                "hf_id": self.tokenizer.hf_id,
                "sampling_rate": self.feature_extractor.sampling_rate,
                "decoder_start_token_id": self.decoder_start_token_id,
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "unk_token_id": self.tokenizer.unk_token_id,
            }
        )
        return config
