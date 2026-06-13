import keras
import numpy as np
from keras import ops

from kerasformers.base import BaseProcessor

from .grounding_dino_image_processor import GroundingDinoImageProcessor
from .grounding_dino_tokenizer import GroundingDinoTokenizer


@keras.saving.register_keras_serializable(package="kerasformers")
class GroundingDinoProcessor(BaseProcessor):
    """Image + text-prompt -> model inputs for Grounding DINO (open-set detection).

    Composes the BERT tokenizer and the DETR-style image processor. ``call``
    lowercases candidate labels and joins them into a single ``". "``-separated
    prompt ending in ``"."`` (the convention the model was trained on), then
    tokenizes and preprocesses the image(s).

    ``post_process`` turns ``logits`` / ``pred_boxes`` into per-image boxes
    (xyxy, scaled to the target size), scores, and the token spans above the
    thresholds.
    """

    TOKENIZER_CLS = GroundingDinoTokenizer
    IMAGE_PROCESSOR_CLS = GroundingDinoImageProcessor
    COMPONENTS = ("tokenizer",)

    def __init__(
        self,
        hf_id=None,
        shortest_edge=800,
        longest_edge=1333,
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge
        self.image_processor = image_processor or GroundingDinoImageProcessor(
            shortest_edge=shortest_edge, longest_edge=longest_edge
        )
        self.tokenizer = tokenizer or GroundingDinoTokenizer(hf_id=hf_id)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def format_text(self, text):
        if isinstance(text, (list, tuple)):
            return ". ".join(t.strip().lower() for t in text) + "."
        text = text.strip().lower()
        return text if text.endswith(".") else text + "."

    def call(self, images=None, text=None):
        if text is None:
            raise ValueError("Grounding DINO requires a `text` prompt.")
        if isinstance(text, str) or (
            isinstance(text, (list, tuple)) and text and isinstance(text[0], str)
        ):
            prompts = [self.format_text(text)]
        else:
            prompts = [self.format_text(t) for t in text]

        tok = self.tokenizer(prompts)
        out = {
            "input_ids": ops.convert_to_tensor(
                np.array(tok["input_ids"], dtype="int64")
            ),
            "attention_mask": ops.convert_to_tensor(
                np.array(tok["attention_mask"], dtype="int64")
            ),
            "token_type_ids": ops.convert_to_tensor(
                np.array(tok["token_type_ids"], dtype="int64")
            ),
        }
        if images is not None:
            img = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(img["pixel_values"])
            out["pixel_mask"] = ops.convert_to_tensor(img["pixel_mask"])
        return out

    def post_process(self, outputs, input_ids, target_sizes, box_threshold=0.3):
        """Decode detector outputs to per-image ``{boxes, scores, token_ids}``.

        Boxes are converted (cx, cy, w, h) -> (x0, y0, x1, y1) and scaled to each
        ``target_sizes`` ``(height, width)``; queries whose max token score
        exceeds ``box_threshold`` are kept.
        """
        logits = np.asarray(ops.convert_to_numpy(outputs["logits"]))
        boxes = np.asarray(ops.convert_to_numpy(outputs["pred_boxes"]))
        ids = np.asarray(ops.convert_to_numpy(ops.convert_to_tensor(input_ids)))
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        scores = probs.max(axis=-1)  # (B, num_queries)
        results = []
        for i, (h, w) in enumerate(target_sizes):
            keep = scores[i] > box_threshold
            cx, cy, bw, bh = (
                boxes[i, :, 0],
                boxes[i, :, 1],
                boxes[i, :, 2],
                boxes[i, :, 3],
            )
            xyxy = np.stack(
                [
                    (cx - bw / 2) * w,
                    (cy - bh / 2) * h,
                    (cx + bw / 2) * w,
                    (cy + bh / 2) * h,
                ],
                axis=-1,
            )
            token_argmax = probs[i].argmax(axis=-1)
            results.append(
                {
                    "boxes": xyxy[keep],
                    "scores": scores[i][keep],
                    "token_ids": [int(ids[i, j]) for j in np.where(keep)[0]]
                    if ids.ndim == 2
                    else [int(token_argmax[j]) for j in np.where(keep)[0]],
                }
            )
        return results

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hf_id": self.hf_id,
                "shortest_edge": self.shortest_edge,
                "longest_edge": self.longest_edge,
            }
        )
        return config
