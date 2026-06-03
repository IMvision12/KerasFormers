import numpy as np
from keras import ops

from kerasformers.base.base_processor import BasePreprocessingLayer
from kerasformers.utils.image_util import get_data_format, load_image


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
    pipelines so processors don't re-derive them: the dtype/channel atoms
    :meth:`to_3_channels` (grayscale/RGBA -> RGB) and :meth:`to_unit_range` (cast +
    0-255 -> [0, 1]); the transform atoms :meth:`rescale` (x * scale), :meth:`resize`,
    :meth:`center_crop` (center crop, zero-padded if smaller), :meth:`pad`,
    :meth:`normalize_image` (data-format-aware ``(x - mean) / std``) and the composite
    :meth:`rescale_and_normalize`; the per-channel normalization constants
    (``OPENAI_CLIP_*``, ``IMAGENET_STANDARD_*``, ``IMAGENET_INCEPTION_*``); and
    :meth:`preprocess_image`, the one-shot ``load -> resize -> rescale -> normalize ->
    transpose`` pipeline (detection / segmentation / depth processors). The per-model
    *resize policy* (aspect-preserving vs square / shortest-edge / smart-resize) stays
    in the concrete processor; only the primitives live here.
    """

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
        image = ops.cast(image, "float32")
        return ops.where(ops.greater(ops.max(image), 1.0), image / 255.0, image)

    @staticmethod
    def normalize_image(x, mean, std, data_format=None):
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
    def rescale(image, scale=1.0 / 255.0):
        return ops.cast(image, "float32") * scale

    @staticmethod
    def resize(image, size, interpolation="bilinear", antialias=True, data_format=None):
        return ops.image.resize(
            image,
            size,
            interpolation=interpolation,
            antialias=antialias,
            data_format=get_data_format(data_format),
        )

    @staticmethod
    def center_crop(image, size, data_format=None):
        data_format = get_data_format(data_format)
        crop_h, crop_w = int(size[0]), int(size[1])
        rank = len(image.shape)
        h_axis, w_axis = (
            (rank - 2, rank - 1)
            if data_format == "channels_first"
            else (rank - 3, rank - 2)
        )
        orig_h, orig_w = int(image.shape[h_axis]), int(image.shape[w_axis])
        top, left = (orig_h - crop_h) // 2, (orig_w - crop_w) // 2

        if orig_h < crop_h or orig_w < crop_w:
            pad_h, pad_w = max(crop_h - orig_h, 0), max(crop_w - orig_w, 0)
            top_pad, left_pad = (pad_h + 1) // 2, (pad_w + 1) // 2  # ceil(pad / 2)
            pad_width = [(0, 0)] * rank
            pad_width[h_axis] = (top_pad, pad_h - top_pad)
            pad_width[w_axis] = (left_pad, pad_w - left_pad)
            image = ops.pad(image, pad_width, constant_values=0)
            top, left = top + top_pad, left + left_pad

        top, left = max(top, 0), max(left, 0)
        starts = [0] * rank
        sizes = [int(d) for d in image.shape]
        starts[h_axis], sizes[h_axis] = top, crop_h
        starts[w_axis], sizes[w_axis] = left, crop_w
        return ops.slice(image, starts, sizes)

    @staticmethod
    def pad(image, padding, mode="constant", constant_values=0.0, data_format=None):
        data_format = get_data_format(data_format)
        if isinstance(padding, int):
            hw = ((padding, padding), (padding, padding))
        elif len(padding) == 2 and isinstance(padding[0], int):
            hw = (tuple(padding), tuple(padding))
        else:
            hw = (tuple(padding[0]), tuple(padding[1]))
        rank = len(image.shape)
        if data_format == "channels_first":
            pad_width = [(0, 0)] * (rank - 2) + [hw[0], hw[1]]
        else:
            pad_width = [(0, 0)] * (rank - 3) + [hw[0], hw[1], (0, 0)]
        if mode == "constant":
            return ops.pad(
                image, pad_width, mode="constant", constant_values=constant_values
            )
        return ops.pad(image, pad_width, mode=mode)

    @staticmethod
    def rescale_and_normalize(
        image,
        do_rescale=True,
        scale=1.0 / 255.0,
        do_normalize=True,
        mean=None,
        std=None,
        data_format=None,
    ):
        if do_rescale:
            image = BaseImageProcessor.rescale(image, scale)
        if do_normalize:
            image = BaseImageProcessor.normalize_image(
                image, mean, std, data_format=data_format
            )
        return image

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
