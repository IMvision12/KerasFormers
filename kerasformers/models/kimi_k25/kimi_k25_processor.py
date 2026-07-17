import keras
import numpy as np

from kerasformers.base import BaseProcessor

from .kimi_k25_image_processor import KimiK25ImageProcessor
from .kimi_k25_tokenizer import KimiK25Tokenizer
from .kimi_k25_video_processor import KimiK25VideoProcessor

IMAGE_TOKEN = "<|media_pad|>"
VIDEO_TOKEN = "<|kimi_k25_video_placeholder|>"
MEDIA_BEGIN = "<|media_begin|>"
MEDIA_CONTENT = "<|media_content|>"
MEDIA_END = "<|media_end|>"


@keras.saving.register_keras_serializable(package="kerasformers")
class KimiK25Processor(BaseProcessor):
    """Text + image + video processor for Kimi K2.5 / K2.6 / K2.7-Code.

    Each ``IMAGE_TOKEN`` in the prompt is expanded to one token per *merged*
    patch (``t * h * w / merge_size**2``); each ``VIDEO_TOKEN`` expands to one
    ``<|media_begin|>video<|media_content|>...<|media_end|>`` span per temporal
    chunk. ``video_token_id`` sits one past the end of the vocabulary, so the
    video runs cannot be tokenized as text -- the prompt is split on
    ``VIDEO_TOKEN`` and the ids are spliced between the encoded segments. The
    model zeroes both placeholders before the embedding lookup and scatters the
    projected patches back in.

    The reference prepends a per-chunk ``hh:mm:ss.fff`` timestamp derived from
    video metadata; frames arrive here already sampled, so no timestamps are
    emitted.

    Args:
        tokenizer / image_processor / video_processor: Pre-built components, or
            omit them to construct the defaults.
        video_token_id: Id spliced in for video patches (163840).
    """

    TOKENIZER_CLS = KimiK25Tokenizer
    IMAGE_PROCESSOR_CLS = KimiK25ImageProcessor
    COMPONENTS = ("tokenizer", "image_processor", "video_processor")

    def __init__(
        self,
        tokenizer=None,
        image_processor=None,
        video_processor=None,
        video_token_id=163840,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or KimiK25Tokenizer()
        self.image_processor = image_processor or KimiK25ImageProcessor()
        self.video_processor = video_processor or KimiK25VideoProcessor()
        self.video_token_id = video_token_id

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(
            tokenizer=KimiK25Tokenizer.from_hf(repo),
            image_processor=KimiK25ImageProcessor.from_hf(repo),
            video_processor=KimiK25VideoProcessor.from_hf(repo),
            **kwargs,
        )

    def merged_tokens(self, grid):
        merge = self.image_processor.merge_size**2
        return int(np.prod(grid)) // merge

    def expand_images(self, text, grids):
        # Split rather than repeated replace(): once the first marker expands, the
        # next replace() would land inside that span instead of the next marker.
        parts = text.split(IMAGE_TOKEN)
        if len(parts) - 1 != len(grids):
            raise ValueError(
                f"{len(parts) - 1} {IMAGE_TOKEN} placeholders but "
                f"{len(grids)} images were given."
            )
        expanded = parts[0]
        for grid, part in zip(grids, parts[1:]):
            expanded += IMAGE_TOKEN * self.merged_tokens(grid) + part
        return expanded

    def video_span(self, grid):
        merge = self.video_processor.merge_size**2
        frame_tokens = int(grid[1]) * int(grid[2]) // merge
        return (
            f"{MEDIA_BEGIN}video{MEDIA_CONTENT}{VIDEO_TOKEN * frame_tokens}{MEDIA_END}"
        )

    def expand_videos(self, text, grids, chunks_per_video):
        parts = text.split(VIDEO_TOKEN)
        if len(parts) - 1 != len(chunks_per_video):
            raise ValueError(
                f"{len(parts) - 1} {VIDEO_TOKEN} placeholders but "
                f"{len(chunks_per_video)} videos were given."
            )
        expanded = parts[0]
        start = 0
        for count, part in zip(chunks_per_video, parts[1:]):
            expanded += "".join(
                self.video_span(g) for g in grids[start : start + count]
            )
            expanded += part
            start += count
        return expanded

    def encode_with_videos(self, text):
        """Encode around the out-of-vocabulary video placeholder."""
        segments = text.split(VIDEO_TOKEN)
        ids = self.tokenizer.encode(segments[0])
        for segment in segments[1:]:
            ids.append(self.video_token_id)
            ids.extend(self.tokenizer.encode(segment))
        return ids

    def call(self, text, images=None, videos=None):
        texts = self.tokenizer.normalize_texts(text)
        inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images)
            inputs.update(image_inputs)
            grids = np.asarray(image_inputs["image_grid_thw"]).tolist()
            per_text = self.deal_per_text(texts, IMAGE_TOKEN, grids)
            texts = [self.expand_images(t, g) for t, g in zip(texts, per_text)]
        if videos is not None:
            video_inputs = self.video_processor(videos)
            chunks = video_inputs.pop("num_chunks_per_video")
            inputs.update(video_inputs)
            grids = np.asarray(video_inputs["video_grid_thw"]).tolist()
            per_text = self.deal_per_text(texts, VIDEO_TOKEN, list(chunks))
            offsets = np.cumsum([0] + [sum(c) for c in per_text]).tolist()
            texts = [
                self.expand_videos(t, grids[offset:], c)
                for t, c, offset in zip(texts, per_text, offsets)
            ]

        sequences = [self.encode_with_videos(t) for t in texts]
        input_ids, attention_mask = self.tokenizer.pad_batch(
            sequences, self.tokenizer.pad_token_id
        )
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"video_token_id": self.video_token_id})
        return config
