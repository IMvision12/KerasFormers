import keras
import numpy as np

from kerasformers.base import BaseImageProcessor, BaseProcessor
from kerasformers.utils.image_util import get_data_format, load_image

from .oneformer_config import ONEFORMER_CONFIG
from .oneformer_tokenizer import OneFormerTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@keras.saving.register_keras_serializable(package="kerasformers")
class OneFormerImageProcessor(BaseImageProcessor):
    """Preprocess images for OneFormer.

    Resizes the longest edge to ``target_size`` (preserving aspect ratio),
    bottom/right-pads to a square ``target_size`` x ``target_size`` canvas,
    rescales to ``[0, 1]``, and applies ImageNet normalization — the same
    fixed-canvas recipe as the Mask2Former processor in this library.

    Args:
        target_size: Target square edge length (matches the model's
            ``image_size``).
        image_mean / image_std: Normalization constants (ImageNet).
        data_format: ``"channels_first"`` / ``"channels_last"``; ``None``
            resolves to ``keras.config.image_data_format()``.
    """

    def __init__(
        self,
        target_size=512,
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_size = target_size
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.data_format = data_format

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
            keras.ops.convert_to_tensor(self.image_mean, dtype="float32"), (1, 1, 1, 3)
        )
        std = keras.ops.reshape(
            keras.ops.convert_to_tensor(self.image_std, dtype="float32"), (1, 1, 1, 3)
        )
        padded = (padded - mean) / std
        if get_data_format(self.data_format) == "channels_first":
            padded = keras.ops.transpose(padded, (0, 3, 1, 2))
        return {"pixel_values": padded}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_size": self.target_size,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class OneFormerProcessor(BaseProcessor):
    """Image + task -> model inputs for OneFormer.

    Combines the image processor with :class:`OneFormerTokenizer`: the chosen
    ``task`` (``"semantic"`` / ``"instance"`` / ``"panoptic"``) is tokenized to
    the ``task_inputs`` float-id vector the model's task MLP consumes, alongside
    the preprocessed ``pixel_values``.

    Args:
        variant: Release variant key (e.g. ``"oneformer_ade20k_swin_tiny"``);
            selects the per-variant tokenizer and the default ``target_size``.
        target_size: Image canvas size (defaults to the variant's
            ``image_size``, else 512).
        task_seq_len: Task prompt length in tokens (77).
        tokenizer: Optional pre-built :class:`OneFormerTokenizer`.
        hf_id: Hub repo to build the tokenizer from (on-the-fly path).
        tokenizer_file: Explicit ``tokenizer.json`` for the tokenizer.
        image_processor: Optional pre-built image processor.
    """

    TOKENIZER_CLS = OneFormerTokenizer
    IMAGE_PROCESSOR_CLS = OneFormerImageProcessor
    COMPONENTS = ("tokenizer",)

    def __init__(
        self,
        variant=None,
        target_size=None,
        task_seq_len=77,
        tokenizer=None,
        hf_id=None,
        tokenizer_file=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.hf_id = hf_id
        if target_size is None:
            target_size = ONEFORMER_CONFIG.get(variant or "", {}).get("image_size", 512)
        self.target_size = target_size
        self.task_seq_len = task_seq_len
        self.image_processor = image_processor or OneFormerImageProcessor(
            target_size=target_size
        )
        self.tokenizer = tokenizer or OneFormerTokenizer(
            variant=variant,
            task_seq_len=task_seq_len,
            hf_id=hf_id,
            tokenizer_file=tokenizer_file,
        )

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def call(self, images=None, task="panoptic"):
        if images is None:
            raise ValueError("Provide `images`.")
        out = self.image_processor(images)
        out["task_inputs"] = keras.ops.convert_to_tensor(
            self.tokenizer.tokenize_task(task)[None]
        )
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "hf_id": self.hf_id,
                "target_size": self.target_size,
                "task_seq_len": self.task_seq_len,
            }
        )
        return config
