# OWLv2 (Scaling open-vocabulary object detection)

**Paper:** [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683) (Minderer et al., 2023)

OWLv2 is the successor to OWL-ViT — same dual-tower (CLIP-style vision
+ text) skeleton, same per-patch detection idea, with two changes that
unlock the larger-scale ("OWL-ST" self-training) recipe:

1. an **objectness head** that scores "is this patch any object at
   all?" independently of the text queries, used at inference to rank
   detection candidates before the text-conditional class score, and
2. a **pad-to-square preprocessing** that preserves the input aspect
   ratio (HF's `Owlv2ImageProcessor` does this; OWL-ViT's processor
   plain-resizes).

## Architecture Highlights

- **Open-vocabulary detection:** classes are arbitrary text strings,
  encoded by the CLIP-style text tower — no fixed softmax head.
- **CLIP-style backbones:** vision tower is a ViT-B/16 or ViT-L/14;
  text tower is a 12-layer GPT-style transformer with a causal mask.
- **Per-patch detection:** every patch token (after class-token
  modulation) emits a box, a per-query class score, and an
  objectness logit.
- **Objectness head (new in v2):** a 3-layer MLP mirroring the box
  head with `out_dim=1` — produces a per-patch objectness logit
  used to filter low-objectness patches at inference.
- **Box bias:** raw box outputs are added to a precomputed per-patch
  log-space bias so each patch's "default" box centers on its grid
  position with a one-patch size.

## Available Variants

| Variant | Vision tower | Image size | HF original |
|---|---|---|---|
| `owlv2-base-patch16` | ViT-B/16 (12L, 768 hidden, 12 heads) | 960×960 | `google/owlv2-base-patch16` |
| `owlv2-base-patch16-ensemble` | ViT-B/16 (12L, 768 hidden, 12 heads) | 960×960 | `google/owlv2-base-patch16-ensemble` |
| `owlv2-base-patch16-finetuned` | ViT-B/16 (12L, 768 hidden, 12 heads) | 960×960 | `google/owlv2-base-patch16-finetuned` |
| `owlv2-large-patch14` | ViT-L/14 (24L, 1024 hidden, 16 heads) | 1008×1008 | `google/owlv2-large-patch14` |
| `owlv2-large-patch14-ensemble` | ViT-L/14 (24L, 1024 hidden, 16 heads) | 1008×1008 | `google/owlv2-large-patch14-ensemble` |
| `owlv2-large-patch14-finetuned` | ViT-L/14 (24L, 1024 hidden, 16 heads) | 1008×1008 | `google/owlv2-large-patch14-finetuned` |

Two classes are exposed:

- `Owlv2Model` — vision + text encoder only (returns `image_embeds`, `text_embeds`).
- `Owlv2Detect` — encoder + class/box/objectness heads for object detection (returns `logits`, `objectness_logits`, `pred_boxes`, …).

The text tower is fixed across variants (12 layers, hidden 512 / 768,
8–16 heads, vocab 49408, max length 16).

## Weights

Pre-converted Keras weights are cached at the `owlv2` GitHub release
on first use:
[https://github.com/IMvision12/KerasFormers/releases/tag/owlv2](https://github.com/IMvision12/KerasFormers/releases/tag/owlv2).
The text-tower BPE vocab (`owlvit_vocab.json` + `owlvit_merges.txt`)
is shared with OWL-ViT and downloaded by `Owlv2Processor` from the
OWL-ViT release.

## End-to-end Example (pure Keras 3, no torch / HF)

```python
from io import BytesIO
import requests
from PIL import Image, ImageDraw

from kerasformers.models.owlv2 import (
    Owlv2Detect,
    Owlv2Processor,
    owlv2_post_process_object_detection,
)

image = Image.open(BytesIO(requests.get(
    "http://images.cocodataset.org/val2017/000000039769.jpg"
).content)).convert("RGB")
text_queries = [["a photo of a cat", "a photo of a dog", "a photo of a remote"]]

processor = Owlv2Processor()
model = Owlv2Detect.from_weights("owlv2-base-patch16-ensemble")

inputs = processor(text=text_queries, images=image)
outputs = model({
    "pixel_values": inputs["pixel_values"],
    "input_ids":    inputs["input_ids"],
})

results = owlv2_post_process_object_detection(
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

image.save("owlv2_output.jpg")
```

## Output Format

The model returns a dict:

| Key | Shape | Description |
|---|---|---|
| `logits` | `(B, num_patches, Q)` | per-query similarity score per patch |
| `objectness_logits` | `(B, num_patches)` | per-patch objectness logit (text-independent) |
| `pred_boxes` | `(B, num_patches, 4)` | normalized `(cx, cy, w, h)` per patch |
| `text_embeds` | `(B, Q, projection_dim)` | L2-normalized text query embeddings |
| `image_embeds` | `(B, h_patches, w_patches, vision_hidden)` | per-patch image features (post `cls`-modulation + LN, shaped as a feature map) |
| `class_embeds` | `(B, num_patches, text_hidden)` | per-patch features projected into the text space |

`B` is the image batch and `Q` is the max number of text queries per
image (the processor flattens text into `B*Q` rows; the model reshapes
back to `(B, Q, ...)` and masks padded queries to `-inf` logits).

`objectness_logits` is the OWLv2-specific output: take `sigmoid(.)`
to get a per-patch "any object" probability, and combine with the
text-conditional `sigmoid(logits)` if you want the joint score.

## Manual Tokenizer / Image Processor Usage

If you want to drive the components separately:

```python
from kerasformers.models.owlv2 import Owlv2Detect, Owlv2ImageProcessor
from kerasformers.models.clip import CLIPTokenizer
from kerasformers.weight_utils import download_file

image_processor = Owlv2ImageProcessor(size={"height": 960, "width": 960})
tokenizer = CLIPTokenizer(
    vocab_file=download_file(
        "https://github.com/IMvision12/KerasFormers/releases/download/owlvit/owlvit_vocab.json"
    ),
    merges_file=download_file(
        "https://github.com/IMvision12/KerasFormers/releases/download/owlvit/owlvit_merges.txt"
    ),
    context_length=16,
    pad_token="!",
)

pixel_values = image_processor(image)["pixel_values"]
text_inputs  = tokenizer(inputs=["a photo of a cat", "a photo of a dog"])

model = Owlv2Detect.from_weights("owlv2-base-patch16-ensemble")
outputs = model({
    "pixel_values": pixel_values,
    "input_ids":    text_inputs["input_ids"],
})
```

## Parity vs HuggingFace Reference

Forward-pass diff between the Keras port (with HF weights) and
``transformers.Owlv2ForObjectDetection`` on the same synthetic
inputs (random RGB image + two text queries):

### `owlv2-base-patch16`

| Output | max_abs_diff |
|---|---:|
| `logits`            | 2.9e-04 |
| `pred_boxes`        | 7.0e-06 |
| `objectness_logits` | 2.3e-04 |
| `logits` cosine sim | 1.0000  |

### `owlv2-base-patch16-ensemble`

| Output | max_abs_diff |
|---|---:|
| `logits`            | 2.4e-04 |
| `pred_boxes`        | 1.8e-05 |
| `objectness_logits` | 2.1e-04 |
| `logits` cosine sim | 1.0000  |

**Status: at fp32 epsilon** — production-ready.

Reproduce on any variant:

```bash
KERAS_BACKEND=torch python -m kerasformers.models.owlv2.convert_owlv2_hf_to_keras
```

## Notes

- **Pad-to-square preprocessing.** Unlike OWL-ViT,
  `Owlv2ImageProcessor` runs `rescale → pad-to-square (zeros) →
  resize → normalize` so the input keeps its aspect ratio.
- **Padded queries.** The tokenizer pads with id 0 (`!`); the class
  predictor uses `input_ids[..., 0] > 0` to mask padded queries to
  `-inf` logits, matching HF.
- **Channels first / last.** The model honors
  `keras.config.image_data_format()`.
- **Box bias.** The per-patch `box_bias` constant is precomputed once
  at model init for the configured grid. Variable-resolution
  inference (HF's `interpolate_pos_encoding=True`) is not currently
  exposed.
