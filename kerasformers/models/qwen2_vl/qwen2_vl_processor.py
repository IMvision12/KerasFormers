"""Qwen2-VL processor — bundles the image processor + tokenizer.

Mirrors HF's ``Qwen2VLProcessor`` API surface: build a ChatML prompt from
messages, run images through the (pure) image processor, expand each
``<|image_pad|>`` placeholder to the right number of merged-patch tokens, and
tokenize. Reused by Qwen2.5-VL (same patch size); Qwen3-VL subclasses with
``patch_size=16``.
"""

import keras
import numpy as np
from keras import ops

from kerasformers.base import BaseProcessor

from .qwen2_vl_image_processor import Qwen2VLImageProcessor
from .qwen2_vl_tokenizer import DEFAULT_TOKENIZER_REPO, Qwen2VLTokenizer

DEFAULT_SYSTEM = "You are a helpful assistant."


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2VLProcessor(BaseProcessor):
    """Image+text -> model inputs for the Qwen-VL models."""

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
        self.tokenizer = Qwen2VLTokenizer(hf_id=hf_id)
        self.image_token = self.tokenizer.image_token

    def apply_chat_template(self, messages, add_generation_prompt=True):
        """Render OpenAI-style ``messages`` to a ChatML prompt string.

        Each ``{"type": "image"}`` content item becomes a single
        ``<|vision_start|><|image_pad|><|vision_end|>`` block (expanded to the
        right token count later, once the image grid is known).
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
                    elif item.get("type") == "text":
                        text += item["text"]
            text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def _expand_image_pads(self, text, grids):
        parts = text.split(self.image_token)
        n_imgs = len(parts) - 1
        if n_imgs != len(grids):
            raise ValueError(
                f"{n_imgs} image placeholders but {len(grids)} images were given."
            )
        m2 = self.spatial_merge_size**2
        out = parts[0]
        for i, g in enumerate(grids):
            n = int(np.prod(g)) // m2
            out += self.image_token * n + parts[i + 1]
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

    def call(
        self,
        conversation=None,
        text=None,
        images=None,
        messages=None,
        add_generation_prompt=True,
    ):
        if conversation is not None:
            messages = conversation
            if images is None:
                images = self._extract_images(conversation)
        if messages is not None:
            text = self.apply_chat_template(messages, add_generation_prompt)
        if text is None:
            raise ValueError("Provide a `conversation`, `messages`, or `text`.")
        texts = [text] if isinstance(text, str) else list(text)

        out = {}
        grids = None
        if images is not None:
            image_inputs = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(image_inputs["pixel_values"])
            out["image_grid_thw"] = ops.convert_to_tensor(
                image_inputs["image_grid_thw"]
            )
            grids = np.asarray(image_inputs["image_grid_thw"])

        if grids is not None:
            texts = [self._expand_image_pads(t, grids) for t in texts]

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
