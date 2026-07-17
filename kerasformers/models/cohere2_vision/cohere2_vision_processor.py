import keras
import numpy as np
from keras import ops

from kerasformers.base import BaseProcessor
from kerasformers.models.cohere2.cohere2_tokenizer import (
    Cohere2Tokenizer,
)

from .cohere2_vision_image_processor import Cohere2VisionImageProcessor


@keras.saving.register_keras_serializable(package="kerasformers")
class Cohere2VisionProcessor(BaseProcessor):
    """Image + text -> model inputs for Cohere2-Vision (Command-A Vision).

    Tiles each image (GotOcr2-style), then expands every ``<image>`` marker to
    one ``image_token`` per merged patch
    (``(size // patch_size // downsample_factor)**2`` per tile) so the model's
    scatter lines up with the projector output. ``call`` accepts a plain
    ``text`` (with ``<image>`` markers) + ``images``.

    Args:
        hf_id: Hub repo for the tokenizer's ``tokenizer.json``.
        size / patch_size / downsample_factor: Tile + projector geometry.
        image_token / image_token_id: Placeholder marker / id.
    """

    TOKENIZER_CLS = Cohere2Tokenizer
    IMAGE_PROCESSOR_CLS = Cohere2VisionImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        hf_id="CohereLabs/command-a-vision-07-2025",
        size=512,
        patch_size=16,
        downsample_factor=2,
        image_token="<image>",
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.size = size
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.image_token = image_token
        self.tokens_per_tile = (size // patch_size // downsample_factor) ** 2
        self.image_processor = image_processor or Cohere2VisionImageProcessor(size=size)
        self.tokenizer = tokenizer or Cohere2Tokenizer(hf_id=hf_id)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def call(self, text=None, images=None):
        if text is None:
            raise ValueError("Provide `text` (with `<image>` markers).")
        texts = [text] if isinstance(text, str) else list(text)
        out = {}
        if images is not None:
            image_inputs = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(image_inputs["pixel_values"])
            num_patches = list(image_inputs["num_patches"])
            # deal_per_text gives each prompt only the tile counts its own markers
            # claim (so a batch does not reuse the first image's geometry) and
            # raises when the markers and images do not add up.
            per_text = self.deal_per_text(texts, self.image_token, num_patches)
            expanded = []
            for t, mine in zip(texts, per_text):
                parts = t.split(self.image_token)
                buf = parts[0]
                for tiles, part in zip(mine, parts[1:]):
                    buf += self.image_token * (self.tokens_per_tile * int(tiles)) + part
                expanded.append(buf)
            texts = expanded
        ids = [self.tokenizer.encode(t) for t in texts]
        max_len = max(len(x) for x in ids)
        input_ids = np.zeros((len(ids), max_len), dtype="int32")
        attention_mask = np.zeros((len(ids), max_len), dtype="int32")
        for i, seq in enumerate(ids):
            input_ids[i, : len(seq)] = seq
            attention_mask[i, : len(seq)] = 1
        out["input_ids"] = ops.convert_to_tensor(input_ids)
        out["attention_mask"] = ops.convert_to_tensor(attention_mask)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hf_id": self.hf_id,
                "size": self.size,
                "patch_size": self.patch_size,
                "downsample_factor": self.downsample_factor,
                "image_token": self.image_token,
            }
        )
        return config
