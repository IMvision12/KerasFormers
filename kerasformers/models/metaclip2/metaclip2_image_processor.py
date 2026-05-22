from typing import Any

import keras
import numpy as np
from keras import ops
from PIL import Image

from kerasformers.models.clip.clip_image_processor import CLIPImageProcessor
from kerasformers.utils.image import load_image


@keras.saving.register_keras_serializable(package="kerasformers")
class MetaClip2ImageProcessor(CLIPImageProcessor):
    """Image processor for MetaCLIP 2 — direct square resize, no center crop.

    Subclass of :class:`CLIPImageProcessor` configured to match the reference
    MetaCLIP 2's default preprocessing. Two differences from OpenAI
    CLIP's processor:

    1. **Direct square resize** to ``(image_resolution, image_resolution)``
       — the reference uses ``do_resize=True, do_center_crop=False``, which means
       the image is stretched (not aspect-preserving) to the target
       square. OpenAI CLIP instead does shortest-edge resize +
       center-crop, which preserves aspect ratio.
    2. **PIL.BICUBIC resample** explicitly — matches the reference's
       ``resample=3`` (``PIL.Image.BICUBIC``) bit-close. The parent
       :class:`CLIPImageProcessor` uses Keras image-ops resize which
       can drift from the reference.

    Pixel values are then rescaled to ``[0, 1]`` and normalized with
    the OpenAI-CLIP mean / std (MetaCLIP 2 keeps these unchanged from
    CLIP).

    Args:
        image_resolution: Target square resolution. Defaults to ``224``.
        mean: Per-channel mean for normalization. Defaults to OpenAI
            CLIP's ``(0.48145466, 0.4578275, 0.40821073)``.
        std: Per-channel std for normalization. Defaults to OpenAI
            CLIP's ``(0.26862954, 0.26130258, 0.27577711)``.
        do_normalize: Whether to apply mean/std normalization.
            Defaults to ``True``.
        do_resize: Whether to resize images to ``image_resolution``.
            Defaults to ``True``.
        data_format: ``"channels_last"`` / ``"channels_first"`` /
            ``None`` (auto from ``keras.config.image_data_format()``).
        **kwargs: Forwarded to :class:`CLIPImageProcessor`.

    Example:
        >>> from kerasformers.models.metaclip2 import (
        ...     MetaClip2ImageProcessor, MetaClip2ZeroShotClassify,
        ... )
        >>> processor = MetaClip2ImageProcessor(image_resolution=224)
        >>> inputs = processor("photo.jpg")
        >>> inputs["pixel_values"].shape   # (1, 224, 224, 3) — channels_last
    """

    def __init__(
        self,
        image_resolution: int = 224,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        do_normalize: bool = True,
        do_resize: bool = True,
        data_format=None,
        **kwargs,
    ):
        super().__init__(
            image_resolution=image_resolution,
            mean=list(mean),
            std=list(std),
            do_center_crop=False,
            do_normalize=do_normalize,
            do_resize=do_resize,
            data_format=data_format,
            **kwargs,
        )

    def process_path(self, image_path: str) -> Any:
        arr = load_image(image_path)
        if self.do_resize:
            pil = Image.fromarray(arr.astype(np.uint8))
            pil = pil.resize(
                (self.image_resolution, self.image_resolution), Image.BICUBIC
            )
            arr = np.array(pil)
        image = arr.astype(np.float32) * np.float32(1.0 / 255.0)
        image = ops.convert_to_tensor(image, dtype="float32")
        if self.do_normalize:
            image = (image - self.mean) / self.std
        return image
