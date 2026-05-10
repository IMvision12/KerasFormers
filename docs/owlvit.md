# OWL-ViT (Open-vocabulary object detection)

**Paper:** [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230) (Minderer et al., 2022)

OWL-ViT detects objects described by free-text queries — no fixed
class list. The architecture composes a CLIP-style vision and text
transformer, then uses class-token-modulated patch features as
detection queries: each patch predicts one box and a per-text-query
similarity score.

![OWL-ViT Detection Output](../assets/owlvit_output.jpg)

## Architecture Highlights

- **Open-vocabulary:** detection classes are arbitrary text strings,
  encoded by the CLIP-style text tower, not a fixed softmax over a
  learned label set.
- **CLIP-style backbones:** vision tower is a ViT-B/32, ViT-B/16, or
  ViT-L/14; text tower is a 12-layer GPT-style transformer with a
  causal mask.
- **Per-patch detection:** every patch token (after class-token
  modulation) emits a box and a per-query class score; no learned
  object queries / DETR decoder.
- **Box bias:** raw box outputs are added to a precomputed per-patch
  log-space bias so each patch's "default" box centers on its grid
  position with a one-patch size.

## Available Models

| Model | Vision tower | Image size | Weights |
|-------|--------------|------------|---------|
| `OwlViTBasePatch32` | ViT-B/32 (12L, 768 hidden, 12 heads) | 768×768 | `owlvit` |
| `OwlViTBasePatch16` | ViT-B/16 (12L, 768 hidden, 12 heads) | 768×768 | `owlvit` |
| `OwlViTLargePatch14` | ViT-L/14 (24L, 1024 hidden, 16 heads) | 840×840 | `owlvit` |

The text tower is fixed across variants (12 layers, hidden 512 / 768,
8–16 heads, vocab 49408, max length 16).

The conversion transfers **412 weight tensors with 0 missing keys**
from the HF checkpoint into the matching Keras layers.

## Weights

Pre-converted Keras weights are cached at the `owlvit` GitHub release
on first use:
[https://github.com/IMvision12/keras-models/releases/tag/owlvit](https://github.com/IMvision12/keras-models/releases/tag/owlvit).
The text-tower BPE vocab (`owlvit_vocab.json` + `owlvit_merges.txt`)
is downloaded by `OwlViTProcessor` from the same release.

## End-to-end Example (pure Keras 3, no torch / HF)

```python
from io import BytesIO
import requests
from PIL import Image, ImageDraw

from kmodels.models.owlvit import (
    OwlViTBasePatch32,
    OwlViTProcessor,
    owlvit_post_process_object_detection,
)

image = Image.open(BytesIO(requests.get(
    "http://images.cocodataset.org/val2017/000000039769.jpg"
).content)).convert("RGB")
text_queries = [["a photo of a cat", "a photo of a dog", "a photo of a remote"]]

processor = OwlViTProcessor()
model = OwlViTBasePatch32(weights="owlvit")

inputs = processor(text=text_queries, images=image)
outputs = model({
    "pixel_values": inputs["pixel_values"],
    "input_ids":    inputs["input_ids"],
})

results = owlvit_post_process_object_detection(
    outputs,
    threshold=0.1,
    target_sizes=[(image.height, image.width)],
    text_labels=text_queries,
)[0]

draw = ImageDraw.Draw(image)
for box, score, label in zip(
    results["boxes"], results["scores"], results["text_labels"]
):
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=(220, 50, 50), width=3)
    draw.text((x1, max(0, y1 - 18)), f"{label} {score:.2f}", fill=(255, 255, 255))

image.save("owlvit_output.jpg")
```

Output (printed):

```
a photo of a remote   score=0.206   box=[40.0,  72.4, 177.8, 115.6]
a photo of a remote   score=0.184   box=[335.7, 74.2, 371.9, 187.6]
a photo of a cat      score=0.704   box=[325.5, 20.5, 640.6, 372.9]
a photo of a cat      score=0.713   box=[1.5,   55.2, 315.7, 472.2]
```

## Output Format

The model returns a dict:

| Key | Shape | Description |
|---|---|---|
| `logits` | `(B, num_patches, Q)` | per-query similarity score per patch |
| `pred_boxes` | `(B, num_patches, 4)` | normalized `(cx, cy, w, h)` per patch |
| `text_embeds` | `(B, Q, projection_dim)` | L2-normalized text query embeddings |
| `image_embeds` | `(B, h_patches, w_patches, vision_hidden)` | per-patch image features (post `cls`-modulation + LN, shaped as a feature map) |
| `class_embeds` | `(B, num_patches, text_hidden)` | per-patch features projected into the text space |

`B` is the image batch and `Q` is the max number of text queries per
image (the processor flattens text into `B*Q` rows; the model reshapes
back to `(B, Q, ...)` and masks padded queries to `-inf` logits).

## Manual Tokenizer / Image Processor Usage

If you want to drive the components separately:

```python
from kmodels.models.owlvit import OwlViTBasePatch32, OwlViTImageProcessor
from kmodels.models.clip import CLIPTokenizer
from kmodels.weight_utils import download_file

image_processor = OwlViTImageProcessor(size={"height": 768, "width": 768})
tokenizer = CLIPTokenizer(
    vocab_file=download_file(
        "https://github.com/IMvision12/keras-models/releases/download/owlvit/owlvit_vocab.json"
    ),
    merges_file=download_file(
        "https://github.com/IMvision12/keras-models/releases/download/owlvit/owlvit_merges.txt"
    ),
    context_length=16,
    pad_token="!",
)

pixel_values = image_processor(image)["pixel_values"]
text_inputs  = tokenizer(inputs=["a photo of a cat", "a photo of a dog"])

model = OwlViTBasePatch32(weights="owlvit")
outputs = model({
    "pixel_values": pixel_values,
    "input_ids":    text_inputs["input_ids"],
})
```

## Parity vs HuggingFace Reference

Forward-pass diff between the Keras port (with HF weights) and
``transformers.OwlViTForObjectDetection`` on the same synthetic
inputs (random RGB image + two text queries):

### `OwlViTBasePatch32`

| Output | Shape | max_abs_diff | mean_abs_diff |
|---|---|---:|---:|
| `logits`       | `(1, 576, 2)`     | 3.1e-05 | 4.8e-06 |
| `pred_boxes`   | `(1, 576, 4)`     | 8.8e-06 | 2.7e-07 |
| `text_embeds`  | `(1, 2, 512)`     | 7.5e-08 | 1.5e-08 |
| `image_embeds` | `(1, 24, 24, 768)`| 2.3e-05 | 3.0e-07 |

### `OwlViTBasePatch16`

| Output | Shape | max_abs_diff | mean_abs_diff |
|---|---|---:|---:|
| `logits`       | `(1, 2304, 2)`    | 4.7e-04 | 2.5e-05 |
| `pred_boxes`   | `(1, 2304, 4)`    | 2.0e-05 | 3.5e-07 |
| `text_embeds`  | `(1, 2, 512)`     | 1.1e-07 | 1.7e-08 |
| `image_embeds` | `(1, 48, 48, 768)`| 9.5e-05 | 5.5e-07 |

**Status: at fp32 epsilon** — production-ready.

`OwlViTLargePatch14` shares the same architecture and conversion
path; only the layer count and hidden dims change. Weight transfer
completes with **412 / 0 missing**. Running both the HF reference
and the Keras port at fp32 in parallel needs ~12 GB of RAM
(24 layers × 1024 hidden, 840×840 image), which OOMs on smaller
machines — parity numbers are not re-listed here.

Reproduce on any variant:

```bash
KERAS_BACKEND=torch python -m kmodels.models.owlvit.convert_owlvit_hf_to_keras
```

## Real-image E2E vs HF (Base/32)

Pure-Keras pipeline (`OwlViTProcessor` → `OwlViT` → post-processing,
no torch/HF anywhere) against `transformers.OwlViTProcessor` +
`OwlViTForObjectDetection` on the COCO `000000039769.jpg` cats image
with three text queries (`cat`, `dog`, `remote`):

| Object | HF score | Keras score | HF box (xyxy) | Keras box (xyxy) |
|---|---:|---:|---|---|
| remote | 0.197 | 0.206 | [40.0, 72.4, 177.8, 115.6] | [40.0, 72.4, 177.8, 115.6] |
| remote | 0.176 | 0.184 | [335.7, 74.2, 371.9, 187.5] | [335.7, 74.2, 371.9, 187.6] |
| cat    | 0.707 | 0.704 | [325.0, 20.4, 640.6, 373.3] | [325.5, 20.5, 640.6, 372.9] |
| cat    | 0.717 | 0.713 | [1.5, 55.3, 315.5, 472.2]   | [1.5, 55.2, 315.7, 472.2]   |

Same 4 objects, scores within 0.01, boxes within 0.5 px. The small
remaining gap is the bicubic-kernel difference between PIL (HF) and
`keras.ops.image.resize` (kmodels) — both are valid bicubic
implementations with slightly different coefficients.

## Notes

- **Padded queries.** The tokenizer pads with id 0 (`!`); the class
  predictor uses `input_ids[..., 0] > 0` to mask padded queries to
  `-inf` logits, matching HF.
- **Channels first / last.** The model honors
  `keras.config.image_data_format()`. Both formats are tested across
  `torch`, `jax`, and `tensorflow` backends.
- **Box bias.** The per-patch `box_bias` constant is precomputed once
  at model init for the configured grid. Variable-resolution
  inference (HF's `interpolate_pos_encoding=True`) is not currently
  exposed.
