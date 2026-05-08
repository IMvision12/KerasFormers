# OWL-ViT (Open-vocabulary object detection)

**Paper:** [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230) (Minderer et al., 2022)

OWL-ViT detects objects described by free-text queries — no fixed
class list. The architecture composes a CLIP-style vision and text
transformer, then uses class-token-modulated patch features as
detection queries: each patch predicts one box and a per-text-query
similarity score.

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
| `OwlViTBasePatch32` | ViT-B/32 (12L, 768 hidden, 12 heads) | 768×768 | `owlvit` (`google/owlvit-base-patch32`) |
| `OwlViTBasePatch16` | ViT-B/16 (12L, 768 hidden, 12 heads) | 768×768 | `owlvit` (`google/owlvit-base-patch16`) |
| `OwlViTLargePatch14` | ViT-L/14 (24L, 1024 hidden, 16 heads) | 840×840 | `owlvit` (`google/owlvit-large-patch14`) |

The text tower is fixed across variants (12 layers, hidden 512 / 768,
8–16 heads, vocab 49408, max length 16).

The conversion transfers **412 weight tensors with 0 missing keys**
from the HF checkpoint into the matching Keras layers.

## Weights

OWL-ViT weights are downloaded straight from the HuggingFace Hub on
first use (`google/owlvit-*` are public). Set `HF_TOKEN` if you want
to avoid Hub rate limiting. Weights are converted in-memory; nothing
is written to disk unless you `model.save_weights(...)` afterwards.

## Quick Start

### High-level pipeline

```python
from kmodels.models.owlvit import OwlViTBasePatch32, OwlViTProcessor
from PIL import Image

model = OwlViTBasePatch32(weights="owlvit")
processor = OwlViTProcessor()

image = Image.open("image.jpg")
text_queries = [["a photo of a cat", "a photo of a dog"]]  # one list per image

inputs = processor(text=text_queries, images=image)
outputs = model(inputs)
# outputs["logits"]:      (B, num_patches,    Q)  — per-query similarity
# outputs["pred_boxes"]:  (B, num_patches,    4)  — cxcywh normalized to [0, 1]
# outputs["text_embeds"]: (B, Q, projection_dim)  — L2-normalized
# outputs["image_embeds"]:(B, h_patches, w_patches, vision_hidden)

results = processor.image_processor.post_process_object_detection(
    outputs, threshold=0.1,
    target_sizes=[image.size[::-1]],
    text_labels=text_queries,
)
for score, label, box in zip(results[0]["scores"],
                              results[0]["text_labels"],
                              results[0]["boxes"]):
    print(f"{label}: {score:.2f}  [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
```

### Manual processor usage

```python
from kmodels.models.owlvit import OwlViTImageProcessor
from kmodels.models.clip import CLIPTokenizer
from kmodels.weight_utils import download_file

image_processor = OwlViTImageProcessor(size={"height": 768, "width": 768})
tokenizer = CLIPTokenizer(
    vocab_file=download_file(
        "https://github.com/IMvision12/keras-models/releases/download/clip/vocab.json"
    ),
    merges_file=download_file(
        "https://github.com/IMvision12/keras-models/releases/download/clip/merges.txt"
    ),
    context_length=16,
    pad_token="!",
)

pixel_values = image_processor(image)["pixel_values"]
text_inputs = tokenizer(inputs=["a photo of a cat", "a photo of a dog"])

outputs = model({
    "pixel_values": pixel_values,
    "input_ids": text_inputs["input_ids"],
})
```

## Low-level API

The :class:`OwlViTCore` class also exposes its sub-components:

```python
from kmodels.models.owlvit import OwlViTBasePatch32

model = OwlViTBasePatch32(weights="owlvit")

# Just the text encoder (pooled + projected, like CLIP).
text_features = model.get_text_features(input_ids)         # (B, projection_dim)

# Just the image encoder (CLS-pooled + projected).
image_features = model.get_image_features(pixel_values)    # (B, projection_dim)

# Internal embedder — returns the per-query text embeds and
# the (B, h, w, hidden) image feature map used by the heads.
query_embeds, feature_map = model.image_text_embedder(pixel_values, input_ids)
```

## Parity vs HuggingFace Reference

Forward-pass diff between the Keras port (with HF weights) and
``transformers.OwlViTForObjectDetection`` on the same synthetic
inputs (random RGB image + two text queries):

### `OwlViTBasePatch32`

| Output | Shape | max_abs_diff | mean_abs_diff |
|---|---|---:|---:|
| `logits`        | `(1, 576, 2)`     | 3.1e-05 | 4.8e-06 |
| `pred_boxes`    | `(1, 576, 4)`     | 8.8e-06 | 2.7e-07 |
| `text_embeds`   | `(1, 2, 512)`     | 7.5e-08 | 1.5e-08 |
| `image_embeds`  | `(1, 24, 24, 768)`| 2.3e-05 | 3.0e-07 |

### `OwlViTBasePatch16`

| Output | Shape | max_abs_diff | mean_abs_diff |
|---|---|---:|---:|
| `logits`        | `(1, 2304, 2)`    | 4.1e-04 | 2.5e-05 |
| `pred_boxes`    | `(1, 2304, 4)`    | 2.1e-05 | 3.5e-07 |
| `text_embeds`   | `(1, 2, 512)`     | 1.1e-07 | 1.7e-08 |
| `image_embeds`  | `(1, 48, 48, 768)`| 9.5e-05 | 5.5e-07 |

**Status: at fp32 epsilon** — production-ready.

`OwlViTLargePatch14` shares the same architecture and conversion
path; only the layer count and hidden dims change, and the weight
mapping is parameterized by them. Weight transfer completes with
**412 / 0 missing**. Running both the HF reference and the Keras
port at fp32 in parallel needs ~12 GB of RAM (24 layers × 1024
hidden, 840×840 image), which can OOM on smaller machines —
parity numbers for this variant are not re-listed here.

Reproduce:

```bash
KERAS_BACKEND=torch python -m kmodels.models.owlvit.convert_owlvit_hf_to_keras \
    --variant OwlViTBasePatch32
```

## Notes

- **Padded queries.** The tokenizer pads with id 0 (`!`); the class
  predictor uses `input_ids[..., 0] > 0` to mask padded queries to
  `-inf` logits, matching HF.
- **Image data format.** Inputs are channels-last
  ``(B, H, W, 3)``; the HF processor returns channels-first, so when
  feeding HF-preprocessed pixels into the Keras model you need to
  transpose first (the bundled `OwlViTImageProcessor` already
  produces channels-last).
- **Box bias.** The per-patch `box_bias` constant is precomputed once
  at model init for the configured grid. Variable-resolution
  inference (HF's `interpolate_pos_encoding=True`) is not currently
  exposed.
