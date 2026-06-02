import keras
import numpy as np
from keras import ops

from kerasformers.base import BaseProcessor
from kerasformers.utils.video_util import load_video

from .qwen2_vl_image_processor import Qwen2VLImageProcessor
from .qwen2_vl_tokenizer import DEFAULT_TOKENIZER_REPO, Qwen2VLTokenizer
from .qwen2_vl_video_processor import Qwen2VLVideoProcessor

DEFAULT_SYSTEM = "You are a helpful assistant."


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2VLProcessor(BaseProcessor):
    """Image / video + text -> model inputs for the Qwen-VL models.

    Composes the tokenizer, image processor, and video processor. ``call``
    renders the chat template, runs the image / video processors, expands each
    ``<|image_pad|>`` / ``<|video_pad|>`` placeholder to the right number of
    merged vision tokens, and tokenizes. A ``{"type": "video"}`` content item (or
    the ``videos=`` argument) yields ``pixel_values_videos`` / ``video_grid_thw``;
    each video is a ``(num_frames, H, W, C)`` array or a list of frames.
    """

    video_processor_cls = Qwen2VLVideoProcessor

    def __init__(
        self,
        hf_id=DEFAULT_TOKENIZER_REPO,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.image_processor = Qwen2VLImageProcessor(
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
        )
        self.video_processor = self.video_processor_cls(
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
        )
        self.tokenizer = Qwen2VLTokenizer(hf_id=hf_id)
        self.image_token = self.tokenizer.image_token
        self.video_token = self.tokenizer.video_token

    def apply_chat_template(self, messages, add_generation_prompt=True):
        """Render OpenAI-style ``messages`` to a ChatML prompt string.

        Each ``{"type": "image"}`` / ``{"type": "video"}`` content item becomes a
        single ``<|vision_start|>`` + ``<|image_pad|>`` / ``<|video_pad|>`` +
        ``<|vision_end|>`` block (expanded to the right token count later, once
        the image / video grid is known).
        """
        has_system = any(m.get("role") == "system" for m in messages)
        text = ""
        if not has_system:
            text += f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n"
            if isinstance(content, str):
                text += content
            else:
                for item in content:
                    if item.get("type") == "image":
                        text += "<|vision_start|><|image_pad|><|vision_end|>"
                    elif item.get("type") == "video":
                        text += "<|vision_start|><|video_pad|><|vision_end|>"
                    elif item.get("type") == "text":
                        text += item["text"]
            text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def _expand_pads(self, text, token, grids):
        """Replace each single ``token`` placeholder with ``prod(grid) // merge^2``
        copies (the number of merged vision tokens for that image / video)."""
        parts = text.split(token)
        n = len(parts) - 1
        if n != len(grids):
            raise ValueError(
                f"{n} {token} placeholders but {len(grids)} vision inputs were given."
            )
        m2 = self.spatial_merge_size**2
        out = parts[0]
        for i, g in enumerate(grids):
            out += token * (int(np.prod(g)) // m2) + parts[i + 1]
        return out

    def _load_image(self, item):
        """Resolve an image content item to a PIL image (path / url / inline)."""
        from PIL import Image

        if item.get("image") is not None:
            return item["image"]
        if item.get("path") is not None:
            return Image.open(item["path"])
        if item.get("url") is not None:
            import io
            import urllib.request

            with urllib.request.urlopen(item["url"]) as resp:
                return Image.open(io.BytesIO(resp.read()))
        raise ValueError("Image content item needs a 'path', 'url', or 'image'.")

    def _load_video(self, item):
        """Resolve a video content item to ``(frames, metadata)``.

        Inline frames come from ``video`` / ``frames``; a ``path`` / ``url`` (or a
        directory of frames / raw bytes) is decoded by
        :func:`kerasformers.utils.video_util.load_video` (PyAV backend) and carries the
        source fps in ``metadata`` so the video processor can subsample to its
        target fps. Like HF, the user only has to point at the file.
        """
        if item.get("video") is not None:
            return item["video"], {"fps": item.get("fps")}
        if item.get("frames") is not None:
            return item["frames"], {"fps": item.get("fps")}
        src = item.get("path") or item.get("url")
        if src is not None:
            frames, metadata = load_video(src, backend="pyav")
            return frames, {"fps": metadata.fps}
        raise ValueError(
            "Video content item needs a 'video', 'frames', 'path', or 'url'."
        )

    def _extract_images(self, conversation):
        """Collect inline images from a conversation's content lists, in order."""
        images = []
        for msg in conversation:
            content = msg.get("content")
            if isinstance(content, (list, tuple)):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        images.append(self._load_image(item))
        return images or None

    def _extract_videos(self, conversation):
        """Collect inline videos as ``(frames, metadata)`` tuples, in order."""
        videos = []
        for msg in conversation:
            content = msg.get("content")
            if isinstance(content, (list, tuple)):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        videos.append(self._load_video(item))
        return videos or None

    def call(
        self,
        conversation=None,
        text=None,
        images=None,
        videos=None,
        messages=None,
        add_generation_prompt=True,
    ):
        video_metas = None
        if conversation is not None:
            messages = conversation
            if images is None:
                images = self._extract_images(conversation)
            if videos is None:
                extracted = self._extract_videos(conversation)
                if extracted is not None:
                    videos = [frames for frames, _ in extracted]
                    video_metas = [meta for _, meta in extracted]
        if messages is not None:
            text = self.apply_chat_template(messages, add_generation_prompt)
        if text is None:
            raise ValueError("Provide a `conversation`, `messages`, or `text`.")
        texts = [text] if isinstance(text, str) else list(text)

        out = {}
        image_grids = None
        video_grids = None
        if images is not None:
            image_inputs = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(image_inputs["pixel_values"])
            out["image_grid_thw"] = ops.convert_to_tensor(
                image_inputs["image_grid_thw"]
            )
            image_grids = np.asarray(image_inputs["image_grid_thw"])
        if videos is not None:
            video_inputs = self.video_processor(videos, video_metadata=video_metas)
            out["pixel_values_videos"] = ops.convert_to_tensor(
                video_inputs["pixel_values_videos"]
            )
            out["video_grid_thw"] = ops.convert_to_tensor(
                video_inputs["video_grid_thw"]
            )
            video_grids = np.asarray(video_inputs["video_grid_thw"])

        if image_grids is not None:
            texts = [self._expand_pads(t, self.image_token, image_grids) for t in texts]
        if video_grids is not None:
            texts = [self._expand_pads(t, self.video_token, video_grids) for t in texts]

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
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "temporal_patch_size": self.temporal_patch_size,
            }
        )
        return config
