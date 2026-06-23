# OneFormer (universal image segmentation)

**Paper**: [OneFormer: One Transformer to Rule Universal Image Segmentation](https://arxiv.org/abs/2211.06220)

OneFormer is a **single model with a single set of weights** that performs
semantic, instance, **and** panoptic segmentation — the task is chosen at inference
time by a task token. It extends the Mask2Former-style mask-classification decoder
with a task-conditioned input: a task string (`"the task is panoptic"`, etc.) is
embedded and fed to the transformer decoder, steering the shared queries toward the
requested segmentation type.

## Architecture Highlights

- **Task-conditioned queries:** a task MLP turns the task string's token ids into a
  task token that conditions the object queries — the same weights produce panoptic,
  instance, or semantic outputs depending on this token.
- **Mask2Former-style decoder:** Swin backbone + multi-scale deformable-attention
  pixel decoder + masked-attention transformer decoder, with a shared `decoder_norm`
  and class/mask prediction heads.
- **Mask classification:** the model predicts a set of (class, binary-mask) pairs;
  panoptic/instance/semantic maps are assembled from them in post-processing.

## Available Variants

| Variant | Backbone | Trained on | HF original |
|---|---|---|---|
| `oneformer_ade20k_swin_tiny` | Swin-Tiny | ADE20K | `shi-labs/oneformer_ade20k_swin_tiny` |
| `oneformer_ade20k_swin_large` | Swin-Large | ADE20K | `shi-labs/oneformer_ade20k_swin_large` |
| `oneformer_coco_swin_large` | Swin-Large | COCO | `shi-labs/oneformer_coco_swin_large` |
| `oneformer_cityscapes_swin_large` | Swin-Large | Cityscapes | `shi-labs/oneformer_cityscapes_swin_large` |

Two classes are exposed:

- `OneFormerModel` — backbone + pixel/transformer decoder (raw features).
- `OneFormerUniversalSegment` — adds the class/mask heads + the weights registry.

## Weights

Pre-converted Keras weights are cached from the `oneformer` GitHub release on first
use:
[https://github.com/IMvision12/KerasFormers/releases/tag/oneformer](https://github.com/IMvision12/KerasFormers/releases/tag/oneformer).
Arbitrary HF fine-tunes (`model_type == "oneformer"`) also load via
`from_weights("hf:<repo>")`.

## Basic Usage

```python
from kerasformers.models.oneformer import OneFormerUniversalSegment

model = OneFormerUniversalSegment.from_weights("oneformer_ade20k_swin_tiny")

# original checkpoint from the Hub
model = OneFormerUniversalSegment.from_weights("hf:shi-labs/oneformer_ade20k_swin_tiny")

# untrained
model = OneFormerUniversalSegment.from_weights("oneformer_ade20k_swin_tiny", load_weights=False)
```

## Inference Example

The processor takes the image **and** the task string; the same model + weights
switch behavior with `task`:

```python
from PIL import Image
from kerasformers.models.oneformer import (
    OneFormerUniversalSegment,
    OneFormerProcessor,
)

model = OneFormerUniversalSegment.from_weights("oneformer_ade20k_swin_tiny")
processor = OneFormerProcessor.from_weights("oneformer_ade20k_swin_tiny")

image = Image.open("image.jpg").convert("RGB")

# task is one of "panoptic" | "instance" | "semantic"
inputs = processor(images=image, task="panoptic")
# inputs: {"pixel_values", "task_inputs"}  (task_inputs is the task MLP's token vector)

output = model(inputs, training=False)
# output["class_queries_logits"]: (1, num_queries, num_labels + 1)
# output["masks_queries_logits"]:  (1, num_queries, H/4, W/4)
```

Each query yields a class distribution and a binary mask; the panoptic / instance /
semantic map is assembled from these pairs (drop the no-object class, match masks to
labels, and resize to the original resolution) — the same mask-classification
post-processing as MaskFormer / Mask2Former.

## Parity vs HuggingFace Reference

Validated against `transformers` (latest main): tiny-config max|Δ| `6e-8`; on the
real `oneformer_ade20k_swin_tiny` weights the segmentation logits match at cosine
`1.0`. Reproduce with:

```bash
KERAS_BACKEND=torch python -m kerasformers.models.oneformer.convert_oneformer_hf_to_keras
```
