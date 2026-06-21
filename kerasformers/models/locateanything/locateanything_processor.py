import keras
import numpy as np
from keras import ops

from kerasformers.base import BaseProcessor

from .locateanything_image_processor import LocateAnythingImageProcessor
from .locateanything_tokenizer import LocateAnythingTokenizer

DEFAULT_SYSTEM = "You are a helpful assistant."


@keras.saving.register_keras_serializable(package="kerasformers")
class LocateAnythingProcessor(BaseProcessor):
    """Image + text -> model inputs for LocateAnything-3B.

    Composes the tokenizer and the native-resolution MoonViT image processor.
    ``call`` renders the ChatML template (each image content item is one
    ``<IMG_CONTEXT>`` placeholder), preprocesses the images to get each one's
    patch grid, expands every placeholder to ``<img>`` +
    ``<IMG_CONTEXT>`` x (``h*w // merge**2``) + ``</img>`` (so the count matches
    MoonViT's merged-token output), and tokenizes to padded
    ``{"input_ids", "attention_mask"}`` alongside ``pixel_values`` /
    ``image_grid_hws``. ``parse_boxes`` decodes generated ids to boxes.
    """

    TOKENIZER_CLS = LocateAnythingTokenizer
    IMAGE_PROCESSOR_CLS = LocateAnythingImageProcessor
    COMPONENTS = ("tokenizer",)

    def __init__(
        self,
        hf_id=None,
        tokenizer=None,
        image_processor=None,
        merge_kernel_size=(2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.image_processor = image_processor or LocateAnythingImageProcessor()
        self.tokenizer = tokenizer or LocateAnythingTokenizer(hf_id=hf_id)
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.image_token = self.tokenizer.image_token
        self.image_start_token = self.tokenizer.image_start_token
        self.image_end_token = self.tokenizer.image_end_token

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def apply_chat_template(
        self, messages, add_generation_prompt=True, system=DEFAULT_SYSTEM
    ):
        text = ""
        if system is not None and not any(m.get("role") == "system" for m in messages):
            text += f"<|im_start|>system\n{system}<|im_end|>\n"
        for msg in messages:
            text += f"<|im_start|>{msg['role']}\n"
            content = msg["content"]
            if isinstance(content, str):
                text += content
            else:
                for item in content:
                    if item.get("type") == "image" or "image" in item:
                        text += self.image_token
                    elif item.get("type") == "text" or "text" in item:
                        text += item.get("text", "")
            text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def expand_image_tokens(self, text, grid_hws):
        parts = text.split(self.image_token)
        n = len(parts) - 1
        if n != len(grid_hws):
            raise ValueError(
                f"{n} image placeholders but {len(grid_hws)} images were given."
            )
        kh, kw = self.merge_kernel_size
        out = parts[0]
        for i, (h, w) in enumerate(grid_hws):
            num_tokens = (int(h) * int(w)) // (kh * kw)
            block = (
                f"<image {i + 1}>"
                + self.image_start_token
                + self.image_token * num_tokens
                + self.image_end_token
            )
            out += block + parts[i + 1]
        return out

    def load_image(self, item):
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

    def extract_images(self, conversation):
        images = []
        for msg in conversation:
            content = msg.get("content")
            if isinstance(content, (list, tuple)):
                for item in content:
                    if isinstance(item, dict) and (
                        item.get("type") == "image" or "image" in item
                    ):
                        images.append(self.load_image(item))
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
                images = self.extract_images(conversation)
        if messages is not None:
            text = self.apply_chat_template(messages, add_generation_prompt)
        if text is None:
            raise ValueError("Provide a `conversation`, `messages`, or `text`.")
        texts = [text] if isinstance(text, str) else list(text)

        out = {}
        if images is not None:
            image_inputs = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(image_inputs["pixel_values"])
            out["image_grid_hws"] = ops.convert_to_tensor(
                image_inputs["image_grid_hws"]
            )
            grid = [tuple(g) for g in np.asarray(image_inputs["image_grid_hws"])]
            texts = [self.expand_image_tokens(t, grid) for t in texts]

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

    def parse_boxes(self, ids):
        return self.tokenizer.parse_boxes(ids)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hf_id": self.hf_id, "merge_kernel_size": list(self.merge_kernel_size)}
        )
        return config
