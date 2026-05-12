# EoMT

**Paper**: [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)

EoMT (Encoder-only Mask Transformer) is a universal segmentation model that simplifies the standard encoder-decoder mask transformer pipeline by using only an encoder architecture. Learned object queries are injected into the final `num_blocks` encoder layers, enabling joint self-attention between image patch tokens and query tokens; after the encoder, query tokens are projected to class logits and mask logits are computed as a bilinear product of query mask embeddings and spatially upscaled patch features.

Two classes are exposed:

- `EoMTModel` — DINOv2-style ViT encoder with the query-injection stack (no task heads). Outputs the post-LayerNorm sequence.
- `EoMTUniversalSegment` — full universal-segmentation model that adds the class predictor, mask head, and mask-feature upscale stack. Output is a dict with `class_logits` and `mask_logits`.

## Architecture Highlights

- **Encoder-Only Design:** Simplifies the standard encode-decode vision pipeline by utilizing a highly efficient encoder-only design for universal segmentation.
- **Unified Segmentation Modeling:** Simultaneously excels at semantic, instance, and panoptic segmentation under one streamlined framework.
- **High Efficiency:** Eliminates the heavyweight decoder yielding drastically improved computational overhead and low latency.

## Available Weights

Pretrained weights are loaded via `EoMTUniversalSegment.from_weights(variant_id)` for kmodels releases, or `EoMTUniversalSegment.from_weights("hf:<repo>")` for arbitrary HF fine-tunes.

| Variant                              | Size  | Dataset            | Classes | Queries | Input    |
|--------------------------------------|-------|--------------------|--------:|--------:|----------|
| `eomt_small_coco_panoptic_640`       | Small | COCO panoptic      |     133 |     200 | 640×640  |
| `eomt_base_coco_panoptic_640`        | Base  | COCO panoptic      |     133 |     200 | 640×640  |
| `eomt_large_coco_panoptic_640`       | Large | COCO panoptic      |     133 |     200 | 640×640  |
| `eomt_large_coco_instance_640`       | Large | COCO instance      |      80 |     200 | 640×640  |
| `eomt_large_ade20k_semantic_512`     | Large | ADE20K semantic    |     150 |     100 | 512×512  |

## Basic Usage

```python
from kmodels.models.eomt import EoMTUniversalSegment

# Small variant (panoptic, COCO)
model = EoMTUniversalSegment.from_weights("eomt_small_coco_panoptic_640")

# Large variant for instance segmentation
model_large = EoMTUniversalSegment.from_weights("eomt_large_coco_instance_640")

# Build untrained model for fine-tuning
custom = EoMTUniversalSegment.from_weights(
    "eomt_large_ade20k_semantic_512",
    load_weights=False,
    num_labels=12,
)
```

### Loading HF fine-tunes

Any HF repo whose `model_type` is `"eomt"` (the official `tue-mps/...` checkpoints or arbitrary user fine-tunes) can be loaded directly via `from_weights("hf:<repo>")`. The class reads hidden size, num layers, queries, num labels, and image size straight from the HF config.

```python
model = EoMTUniversalSegment.from_weights(
    "hf:tue-mps/coco_panoptic_eomt_large_640"
)
```

## Inference Example

```python
from kmodels.models.eomt import EoMTUniversalSegment, EoMTImageProcessor
from PIL import Image

model = EoMTUniversalSegment.from_weights("eomt_large_coco_panoptic_640")

image = Image.open("image.jpg").convert("RGB")
original_h, original_w = image.size[1], image.size[0]

# Preprocess: resize, pad to square, rescale, ImageNet normalize
processor = EoMTImageProcessor(target_size=640)
inputs = processor(image)

# Inference
output = model(inputs["pixel_values"], training=False)
# output["class_logits"]: (1, num_queries, 134) — class logits per query
# output["mask_logits"]:  (1, num_queries, mask_h, mask_w) — mask logits

#   processor.post_process_panoptic_segmentation(...)
#   processor.post_process_semantic_segmentation(...)
#   processor.post_process_instance_segmentation(...)
result = processor.post_process_panoptic_segmentation(
    output, target_size=(original_h, original_w), threshold=0.8,
)
for seg in result["segments_info"][:6]:
    name = seg["label_name"].replace("things: ", "").replace("stuff: ", "")
    print(f"{name}: {seg['score']:.2f}")

# Output:
# cat: 1.00
# cat: 1.00
# couch: 0.95
# remote: 1.00
# remote: 1.00
```

## Segmentation Modes

`EoMTImageProcessor` exposes the three post-processing modes as methods,
mirroring HuggingFace `transformers`:

```python
processor = EoMTImageProcessor(target_size=640)
output = model(processor(image), training=False)
target_size = (image.height, image.width)

# Panoptic (things + stuff merged into one segmentation)
panoptic = processor.post_process_panoptic_segmentation(
    output, target_size=target_size, threshold=0.8,
)

# Semantic (per-pixel class id, no instance separation)
semantic = processor.post_process_semantic_segmentation(
    output, target_size=target_size,
)

# Instance (per-object binary masks)
instance = processor.post_process_instance_segmentation(
    output, target_size=target_size, threshold=0.5,
)
```

| Method | Returns |
|---|---|
| `post_process_panoptic_segmentation` | `{"segmentation": (H, W) int32, "segments_info": [...]}` |
| `post_process_semantic_segmentation` | `{"segmentation": (H, W) int32, "class_names": [...]}` |
| `post_process_instance_segmentation` | `{"segmentation": (H, W) int32, "segments_info": [...]}` |

The same forward pass on `model` is shared across all three methods —
just call the one(s) you need.

### Data format

Every processor and format-sensitive post-processor in this module accepts a `data_format=None` kwarg. The default (`None`) resolves to `keras.config.image_data_format()`; pass `"channels_first"` or `"channels_last"` to override per-call without touching global state.

```python
# follow the global config (the default)
processor = EoMTImageProcessor()
inputs = processor("photo.jpg")

# force channels_first for this call only
processor = EoMTImageProcessor(data_format="channels_first")
inputs = processor("photo.jpg")
```

Image processors return tensors in the requested layout; post-processors accept tensors in either layout and read the flag to pick the channel axis. See `docs/utils.md` for which families have format-sensitive post-processors.

## Full Inference with Visualization

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kmodels.models.eomt import EoMTUniversalSegment, EoMTImageProcessor

model = EoMTUniversalSegment.from_weights("eomt_large_coco_panoptic_640")

img = Image.open("image.jpg").convert("RGB")
original_h, original_w = img.size[1], img.size[0]

processor = EoMTImageProcessor(target_size=640)
inputs = processor(img)
output = model(inputs["pixel_values"], training=False)

result = processor.post_process_panoptic_segmentation(
    output, target_size=(original_h, original_w), threshold=0.8,
)

segmentation = result["segmentation"]
segments_info = result["segments_info"]

np.random.seed(42)
colors = np.random.randint(50, 220, size=(len(segments_info) + 1, 3), dtype=np.uint8)

colored = np.zeros((original_h, original_w, 3), dtype=np.uint8)
for seg in segments_info:
    mask = segmentation == seg["id"]
    colored[mask] = colors[seg["id"]]

overlay = np.array(img).copy()
alpha = 0.5
has_seg = segmentation >= 0
overlay[has_seg] = (overlay[has_seg] * (1 - alpha) + colored[has_seg] * alpha).astype(np.uint8)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.imshow(overlay)

legend_patches = []
legend_names = []
for seg in segments_info[:10]:
    color = colors[seg["id"]] / 255.0
    patch = plt.Rectangle((0, 0), 1, 1, fc=color)
    legend_patches.append(patch)
    name = seg["label_name"].replace("things: ", "").replace("stuff: ", "")
    legend_names.append(f"{name}: {seg['score']:.2f}")
if legend_patches:
    ax.legend(legend_patches, legend_names, loc="upper right", fontsize=10)

ax.set_title("EoMT Panoptic Segmentation", fontsize=16)
ax.axis("off")
plt.tight_layout()
fig.savefig("eomt_output.jpg", bbox_inches="tight", dpi=120)
plt.close(fig)
```

![EoMT Panoptic Segmentation Output](../assets/eomt_output.jpg)
