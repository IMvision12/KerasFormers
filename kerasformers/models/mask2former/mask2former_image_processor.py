from typing import Optional, Tuple

import keras
import numpy as np

from kerasformers.base import BaseImageProcessor
from kerasformers.utils.image_util import get_data_format, load_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@keras.saving.register_keras_serializable(package="kerasformers")
class Mask2FormerImageProcessor(BaseImageProcessor):
    """Preprocess images for Mask2Former.

    Resizes the longest edge to ``target_size`` (preserving aspect ratio),
    bottom/right-pads to a square ``target_size`` x ``target_size`` canvas,
    rescales to ``[0, 1]``, and applies ImageNet normalization. Uses pure
    Keras 3 ops for all tensor operations, and emits the pixel values in the
    configured data format.

    Args:
        target_size: Target square edge length (matches the model's
            ``image_size``).
        image_mean: Per-channel mean for normalization. Defaults to the
            ImageNet mean.
        image_std: Per-channel standard deviation for normalization. Defaults
            to the ImageNet std.
        data_format: ``"channels_first"`` / ``"channels_last"``; ``None``
            resolves to ``keras.config.image_data_format()``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`BaseImageProcessor`.
    """

    def __init__(
        self,
        target_size: int = 384,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_size = target_size
        self.image_mean = image_mean if image_mean is not None else IMAGENET_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STD
        self.data_format = data_format

    def __call__(self, image):
        return self.call(image)

    def call(self, image):
        if isinstance(image, np.ndarray) and image.ndim == 4:
            image = image[0]
        image = load_image(image).astype(np.float32)

        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        image = keras.ops.convert_to_tensor(image, dtype="float32")
        image = keras.ops.expand_dims(image, axis=0)
        image = keras.ops.image.resize(image, (new_h, new_w), interpolation="bilinear")
        image = image / 255.0

        padded = keras.ops.zeros(
            (1, self.target_size, self.target_size, 3), dtype="float32"
        )
        padded = keras.ops.slice_update(padded, (0, 0, 0, 0), image)

        mean = keras.ops.reshape(
            keras.ops.convert_to_tensor(self.image_mean, dtype="float32"),
            (1, 1, 1, 3),
        )
        std = keras.ops.reshape(
            keras.ops.convert_to_tensor(self.image_std, dtype="float32"),
            (1, 1, 1, 3),
        )
        padded = (padded - mean) / std

        if get_data_format(self.data_format) == "channels_first":
            padded = keras.ops.transpose(padded, (0, 3, 1, 2))

        return {"pixel_values": padded}
