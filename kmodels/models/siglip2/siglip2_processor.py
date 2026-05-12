from typing import List, Optional, Union

import keras

from kmodels.base import BaseProcessor
from kmodels.models.siglip2.siglip2_image_processor import SigLIP2ImageProcessor
from kmodels.models.siglip2.siglip2_tokenizer import SigLIP2Tokenizer
from kmodels.weight_utils import download_file


@keras.saving.register_keras_serializable(package="kmodels")
class SigLIP2Processor(BaseProcessor):
    """Combined processor for SigLIP 2 models — image + Gemma text.

    Pairs :class:`SigLIP2ImageProcessor` (resize / center crop /
    normalize) with :class:`SigLIP2Tokenizer` (Gemma SentencePiece,
    vocab 256000). Downloads the Gemma SP model on first use when
    ``vocab_file`` is not supplied.
    """

    def __init__(
        self,
        image_resolution: int = 224,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        vocab_file: Optional[str] = None,
        context_length: int = 64,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        add_bos: bool = False,
        add_eos: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_processor = SigLIP2ImageProcessor(
            image_resolution=image_resolution,
            mean=mean,
            std=std,
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
        )

        if vocab_file is None:
            vocab_file_path = download_file(
                "https://github.com/IMvision12/keras-models/releases/download/SigLIP/siglip2_vocab.model"
            )
        else:
            vocab_file_path = vocab_file

        self.tokenizer = SigLIP2Tokenizer(
            vocab_file=vocab_file_path,
            context_length=context_length,
            add_bos=add_bos,
            add_eos=add_eos,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )

        self._config = {
            "image_resolution": image_resolution,
            "mean": mean,
            "std": std,
            "do_center_crop": do_center_crop,
            "do_normalize": do_normalize,
            "do_resize": do_resize,
            "vocab_file": vocab_file,
            "context_length": context_length,
            "pad_token": pad_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "unk_token": unk_token,
            "add_bos": add_bos,
            "add_eos": add_eos,
        }

    def call(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[keras.KerasTensor, List]] = None,
        image_paths: Optional[Union[str, List[str]]] = None,
    ):
        if text is None and images is None and image_paths is None:
            raise ValueError(
                "At least one of 'text', 'images', or 'image_paths' must be provided"
            )
        if images is not None and image_paths is not None:
            raise ValueError("Cannot specify both 'images' and 'image_paths'")

        encoding = {}
        if text is not None:
            encoding.update(self.tokenizer(inputs=text))
        if images is not None:
            encoding["images"] = self.image_processor(images)["pixel_values"]
        if image_paths is not None:
            encoding["images"] = self.image_processor(image_paths)["pixel_values"]
        return encoding

    def decode_text(
        self, token_ids: keras.KerasTensor, skip_special_tokens: bool = True
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
