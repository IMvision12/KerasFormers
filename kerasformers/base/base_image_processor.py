import numpy as np
from keras import ops

from kerasformers.base.base_processor import BasePreprocessingLayer
from kerasformers.utils.image import get_data_format, load_image


class BaseImageProcessor(BasePreprocessingLayer):
    """Abstract base for kerasformers image preprocessors.

    Subclasses implement ``call(images)`` returning the model-ready pixel tensor
    (or a dict that includes one). The loading API (``from_weights`` /
    ``from_release`` / ``from_hf``) and the ``__call__`` -> ``call`` forwarder are
    inherited from :class:`BasePreprocessingLayer`. Concrete subclasses define
    their own constructor kwargs (resolution, normalization stats, interpolation
    mode, patch size, …) and ``get_config`` payload — the base bakes in no
    defaults.

    Provides backend-agnostic (``keras.ops``) building blocks shared by the pixel
    pipelines so processors don't re-derive them: the atoms :meth:`to_3_channels`
    (grayscale/RGBA -> RGB) and :meth:`to_unit_range` (cast + 0-255 -> [0, 1]); the
    per-channel normalization constants (``OPENAI_CLIP_*``, ``IMAGENET_STANDARD_*``,
    ``IMAGENET_INCEPTION_*``);
    and the heavier shared pipeline pieces :meth:`normalize_image` (data-format-aware
    normalize, rank 3 or 4) and :meth:`preprocess_image` (one-shot load -> resize ->
    rescale -> normalize -> transpose, used by the detection / segmentation / depth
    processors). Resize / crop atoms stay per-model since they vary (aspect-preserving
    vs square, with/without padding).
    """

    # Common per-channel RGB normalization statistics, as constants so concrete
    # processors reference them instead of repeating the magic numbers.
    OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    IMAGENET_STANDARD_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STANDARD_STD = (0.229, 0.224, 0.225)
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

    def call(self, images):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `call(images)`."
        )

    @staticmethod
    def to_3_channels(image):
        # Expand grayscale (1 -> RGB) / drop alpha (RGBA -> RGB); pass RGB through.
        num_channels = ops.shape(image)[-1]
        if num_channels == 1:
            return ops.repeat(image, 3, axis=-1)
        if num_channels == 4:
            return image[..., :3]
        if num_channels == 3:
            return image
        raise ValueError(f"Unsupported number of image channels: {num_channels}")

    @staticmethod
    def to_unit_range(image):
        # Cast to float32 and bring 0-255 inputs into [0, 1]; inputs already in
        # [0, 1] (max <= 1) pass through unchanged.
        image = ops.cast(image, "float32")
        return ops.where(ops.greater(ops.max(image), 1.0), image / 255.0, image)

    @staticmethod
    def normalize_image(x, mean, std, data_format=None):
        # Data-format-aware (x - mean) / std along the channel axis (rank 3 or 4):
        # reshapes the per-channel stats to the channel position required by
        # `data_format` (pass data_format="channels_last" for HWC inputs).
        data_format = get_data_format(data_format)
        mean = ops.convert_to_tensor(mean, dtype="float32")
        std = ops.convert_to_tensor(std, dtype="float32")
        rank = len(x.shape)
        if data_format == "channels_first":
            shape = (1, -1, 1, 1) if rank == 4 else (-1, 1, 1)
        else:
            shape = (1, 1, 1, -1) if rank == 4 else (1, 1, -1)
        mean = ops.reshape(mean, shape)
        std = ops.reshape(std, shape)
        return (x - mean) / std

    @staticmethod
    def preprocess_image(
        images,
        target_size,
        image_mean=None,
        image_std=None,
        rescale=True,
        interpolation="bilinear",
        antialias=True,
        data_format=None,
    ):
        data_format = get_data_format(data_format)

        if isinstance(images, (list, tuple)):
            items = list(images)
        elif isinstance(images, np.ndarray) and images.ndim == 4:
            items = [images[i] for i in range(images.shape[0])]
        else:
            items = [images]
        if not items:
            raise ValueError("`images` must contain at least one image.")

        loaded = [load_image(img) for img in items]
        original_sizes = [(int(arr.shape[0]), int(arr.shape[1])) for arr in loaded]

        if isinstance(target_size, int):
            target_h = target_w = target_size
        else:
            target_h, target_w = target_size

        per_image = []
        for arr in loaded:
            t = ops.convert_to_tensor(arr, dtype="float32")
            t = ops.expand_dims(t, axis=0)
            t = ops.image.resize(
                t,
                size=(target_h, target_w),
                interpolation=interpolation,
                antialias=antialias,
                data_format="channels_last",
            )
            per_image.append(t)

        x = ops.concatenate(per_image, axis=0)

        if rescale:
            x = x / 255.0

        if image_mean is not None:
            if image_std is None:
                raise ValueError("image_std must be provided when image_mean is set.")
            x = BaseImageProcessor.normalize_image(
                x, image_mean, image_std, data_format="channels_last"
            )

        if data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))

        return x, original_sizes, (target_h, target_w), data_format
