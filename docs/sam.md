# SAM (Segment Anything Model)

## Overview

SAM (Segment Anything Model) is a promptable segmentation model that can generate high-quality segmentation masks for any object in an image, given input prompts such as points, bounding boxes, or masks. It was trained on the SA-1B dataset containing over 1 billion masks on 11 million images.

**Reference:** [Segment Anything](https://arxiv.org/abs/2304.02643) (Kirillov et al., 2023)

Two classes are exposed:

- `SAMVisionModel` — ViT vision encoder + neck (no prompt encoder, no mask decoder). Returns image embeddings ``(B, 64, 64, 256)``. Use this to cache image features when running many prompt combinations.
- `SAMPromptableSegment` — full promptable segmentation model. Composes the vision encoder with the prompt encoder and mask decoder. Returns ``{"pred_masks", "iou_scores"}``.

## Architecture Highlights

- **Promptable Object Segmentation:** Naturally accepts ambiguous or explicit prompts in the form of interactive points, bounding boxes, or dense masks.
- **Zero-Shot Generalization:** Delivers high-quality masks out-of-the-box on novel domains and unseen subjects without retraining.
- **Three-Part Pipeline:** Features a robust ViT Image Encoder, a flexible sparse/dense Prompt Encoder, and a lightweight two-way Mask Decoder for lightning-fast prompting.
- **Ambiguity Awareness:** Generates multiple valid segmentation mask hypotheses when a prompt is underspecified (e.g. part vs whole).

## Available Weights

Pretrained weights are loaded via `SAMPromptableSegment.from_weights(variant_id)` (or `from_hf(hf_id)` for arbitrary HF fine-tunes).

| Variant         | Parameters | Backbone   |
|-----------------|-----------:|------------|
| `sam_vit_base`  |     ~93 M  | ViT-B/16   |
| `sam_vit_large` |    ~308 M  | ViT-L/16   |
| `sam_vit_huge`  |    ~636 M  | ViT-H/16   |

All variants take a 1024×1024 input. The Huge checkpoint is stored as a sharded ``.weights.json`` index + shards on GitHub Releases; loading is transparent.

## Basic Usage

```python
from kmodels.models.sam import SAMPromptableSegment

# Build a SAM model with default 1024x1024 input and multi-mask output
model = SAMPromptableSegment.from_weights("sam_vit_base")

# For single best-mask output instead of 3 ambiguity hypotheses
model_single = SAMPromptableSegment.from_weights(
    "sam_vit_base", multimask_output=False
)

# Build an untrained model
model_init = SAMPromptableSegment.from_weights(
    "sam_vit_base", load_weights=False
)
```

`multimask_output` is a **construction-time** flag in the Keras port (unlike the HuggingFace port where it's a runtime kwarg). The Keras functional graph needs to be built for one mode or the other.

### Loading HF fine-tunes

Any HF repo whose `model_type` is `"sam"` (the official `facebook/sam-vit-*` checkpoints or any user fine-tune built on the same architecture) can be loaded directly with `from_hf`. The class reads ViT dims, MLP dim, and global-attention indices straight from the HF config.

```python
model = SAMPromptableSegment.from_hf("facebook/sam-vit-base")
```

## Model Inputs

The SAM model's functional graph requires `pixel_values`, `input_points`, and `input_labels`. Enable optional prompt kinds at construction time via `enable_boxes=True` / `enable_masks=True`; otherwise those inputs are not part of the graph.

| Input | Shape | Description |
|---|---|---|
| `pixel_values` | `(batch, 1024, 1024, 3)` | Normalized image (ImageNet mean/std). |
| `input_points` | `(batch, point_batch, num_points, 2)` | `(x, y)` pixel coords in the model input frame. |
| `input_labels` | `(batch, point_batch, num_points)` | `1` foreground, `0` background, `-1` padding, `-10` ignore. |
| `input_boxes` (when `enable_boxes=True`) | `(batch, point_batch, 4)` | `(x1, y1, x2, y2)`. Dim‑1 must match `point_batch`. |
| `has_boxes_input` (when `enable_boxes=True`) | `(batch, 1)` | `1.0` if `input_boxes` is meaningful, else `0.0`. |
| `input_masks` (when `enable_masks=True`) | `(batch, 256, 256, 1)` | Dense mask prompt at 4× downscale of the model input. |
| `has_mask_input` (when `enable_masks=True`) | `(batch, 1)` | `1.0` if `input_masks` is meaningful, else `0.0`. |

## Inference with Point Prompts

```python
import numpy as np
import keras
from kmodels.models.sam import (
    SAMPromptableSegment, SAMImageProcessorWithPrompts,
)

model = SAMPromptableSegment.from_weights("sam_vit_base")

processor = SAMImageProcessorWithPrompts(
    input_points=np.array([[[390, 280]]]),  # (x, y) pixel coord on the subject
    input_labels=np.array([[1]]),           # 1 = foreground
)
inputs = processor("groceries.jpg")

outputs = model.predict(
    {
        "pixel_values": inputs["pixel_values"],
        "input_points": inputs["input_points"],
        "input_labels": inputs["input_labels"],
    },
    verbose=0,
)

masks = processor.post_process_masks(
    outputs["pred_masks"],
    original_size=inputs["original_size"],
    reshaped_size=inputs["reshaped_size"],
)

iou_scores = keras.ops.convert_to_numpy(outputs["iou_scores"])[0, 0]
best_idx = int(np.argmax(iou_scores))
best_mask = keras.ops.convert_to_numpy(masks)[0, 0, best_idx] > 0.0
print(f"IoU: {iou_scores[best_idx]:.3f}, Mask shape: {best_mask.shape}")
```

### Data format

Every processor and format-sensitive post-processor in this module accepts a `data_format=None` kwarg. The default (`None`) resolves to `keras.config.image_data_format()`; pass `"channels_first"` or `"channels_last"` to override per-call without touching global state.

```python
# follow the global config (the default)
processor = SAMImageProcessor()
inputs = processor("photo.jpg")

# force channels_first for this call only
processor = SAMImageProcessor(data_format="channels_first")
inputs = processor("photo.jpg")
```

Image processors return tensors in the requested layout; post-processors accept tensors in either layout and read the flag to pick the channel axis. See `docs/utils.md` for which families have format-sensitive post-processors.

## Inference with Box Prompts

Pass a real `(x1, y1, x2, y2)` box and toggle `has_boxes_input=1`. Build the model with `enable_boxes=True` so the box inputs are part of the graph:

```python
import numpy as np
from kmodels.models.sam import (
    SAMPromptableSegment, SAMImageProcessorWithPrompts,
)

model = SAMPromptableSegment.from_weights("sam_vit_base", enable_boxes=True)

processor = SAMImageProcessorWithPrompts(
    input_points=np.zeros((1, 1, 1, 2), dtype="float32"),   # placeholder
    input_labels=-10 * np.ones((1, 1, 1), dtype="int32"),   # ignore
    input_boxes=np.array([[100, 200, 400, 500]]),           # (x1, y1, x2, y2)
)
inputs = processor("photo.jpg")
inputs["has_boxes_input"] = np.ones((1, 1), dtype="float32")

outputs = model.predict(inputs, verbose=0)
masks = processor.post_process_masks(
    outputs["pred_masks"],
    original_size=inputs["original_size"],
    reshaped_size=inputs["reshaped_size"],
)
```

Note the shape constraint: `input_boxes` dim‑1 must equal `point_batch`. For a pure-box prompt, pass a matching zero-point placeholder with `label=-10` (ignore).

## Mask Refinement

Feed a coarse low-resolution mask back in to refine the output. Build the model with `enable_masks=True`; masks must be passed at the 256×256 prompt-encoder resolution:

```python
model = SAMPromptableSegment.from_weights(
    "sam_vit_base", enable_masks=True
)
# ... initial inference returns outputs["pred_masks"]
coarse = outputs["pred_masks"][0, 0, 0:1]                # (1, 256, 256)
coarse = np.transpose(coarse, (1, 2, 0))[None, ...]      # (1, 256, 256, 1)

inputs["input_masks"]    = coarse.astype("float32")
inputs["has_mask_input"] = np.ones((1, 1), dtype="float32")
refined = model.predict(inputs, verbose=0)
```

## Precomputed Image Embeddings (Multi-Prompt Inference)

For interactive tools that try many prompts on the same image, run the ViT encoder **once** and reuse its output. Every `SAMPromptableSegment` instance exposes three sub-models:

| Attribute | Inputs | Outputs |
|---|---|---|
| `model.vision_encoder_model` | `pixel_values` | `image_embeddings` `(1, 64, 64, 256)` |
| `model.prompt_decoder_model` | `image_embeddings` + prompt inputs | `pred_masks`, `iou_scores` |
| `model.prompt_encoder_model` | prompt inputs | `sparse_embeddings`, `dense_embeddings` |

```python
from kmodels.models.sam import SAMPromptableSegment, SAMImageProcessor

model = SAMPromptableSegment.from_weights("sam_vit_base")
processor = SAMImageProcessor()
pre = processor("photo.jpg")

# Run the vision encoder once
image_embeddings = model.get_image_embeddings(pre["pixel_values"])

# Try many prompts without re-running the ViT
for (x, y) in [(450, 600), (200, 150), (700, 300)]:
    out = model.prompt_decoder_model.predict({
        "image_embeddings":  image_embeddings,
        "input_points":      np.array([[[[x, y]]]], dtype="float32"),
        "input_labels":      np.array([[[1]]], dtype="int32"),
    }, verbose=0)
```

For a 1024×1024 image the ViT is roughly 95% of the compute — the decoder itself runs in milliseconds.

Alternatively use `SAMVisionModel` standalone if you only need image embeddings (no prompt encoder / mask decoder built):

```python
from kmodels.models.sam import SAMVisionModel
vision = SAMVisionModel.from_weights("sam_vit_base")
image_embeddings = vision(pre["pixel_values"])
```

You can also extract prompt embeddings without invoking the mask decoder:

```python
pe = model.get_prompt_embeddings(input_points, input_labels)
# pe["sparse_embeddings"]: (batch, point_batch, N, 256)
# pe["dense_embeddings"] : (batch, 64, 64, 256)
```

## Full Inference with Visualization

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kmodels.models.sam import (
    SAMPromptableSegment, SAMImageProcessorWithPrompts,
)

COLORS = [
    np.array([0, 180, 255, 150]) / 255.0,
    np.array([255, 90, 60, 150]) / 255.0,
]


def show_mask(mask, ax, color):
    h, w = mask.shape
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def show_points(coords, ax, color, marker_size=340):
    ax.scatter(coords[:, 0], coords[:, 1], color=color, marker="*",
               s=marker_size, edgecolors="white", linewidths=1.25, zorder=5)


model = SAMPromptableSegment.from_weights("sam_vit_large")
img = Image.open("assets/coco_cats.jpg").convert("RGB")   # COCO val2017/000000039769.jpg

prompts = [
    {"points": np.array([[[150, 200]]]), "labels": np.array([[1]]), "name": "left cat"},
    {"points": np.array([[[440, 180]]]), "labels": np.array([[1]]), "name": "right cat"},
]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.imshow(np.array(img))

for i, prompt in enumerate(prompts):
    processor = SAMImageProcessorWithPrompts(
        input_points=prompt["points"],
        input_labels=prompt["labels"],
    )
    inputs = processor(img)
    outputs = model.predict({
        "pixel_values": inputs["pixel_values"],
        "input_points": inputs["input_points"],
        "input_labels": inputs["input_labels"],
    }, verbose=0)

    masks = processor.post_process_masks(
        outputs["pred_masks"],
        original_size=inputs["original_size"],
        reshaped_size=inputs["reshaped_size"],
    )
    masks_np = keras.ops.convert_to_numpy(masks)[0, 0]
    iou_scores = keras.ops.convert_to_numpy(outputs["iou_scores"])[0, 0]
    best_idx = int(np.argmax(iou_scores))

    color = COLORS[i]
    show_mask(masks_np[best_idx] > 0.0, ax, color)
    show_points(prompt["points"][0], ax, color=color[:3])
    print(f"  {prompt['name']}: IoU={iou_scores[best_idx]:.3f}")

ax.set_title("SAM ViT-Large - Point Prompts (COCO cats)", fontsize=14)
ax.axis("off")
plt.tight_layout()
fig.savefig("sam_train_output.jpg", bbox_inches="tight", dpi=130)
plt.close(fig)
```

![SAM Point Prompts Output](../assets/sam_train_output.jpg)

Running this on the default HF-parity model (three inputs: `pixel_values`, `input_points`, `input_labels`) on the classic two-cats COCO image (``val2017/000000039769.jpg``, saved locally as ``assets/coco_cats.jpg``) segments each cat from a single point click with IoU scores > 0.99.

## Automatic Mask Generation ("Segment Everything")

Without any prompts, SAM can sample a dense point grid over the image and return every mask it can find. The Keras port ships both the HuggingFace-parity helpers and a driver that ties them together.

### What's where

HuggingFace's `SamProcessor` exposes the **helpers** (`generate_crop_boxes`, `filter_masks`, `post_process_for_mask_generation`, plus internals like `_compute_stability_score`, `_mask_to_rle`) but leaves the crop loop, per-crop batching, and model orchestration to you. Meta's original `segment-anything` repo ships the end-to-end driver as `SamAutomaticMaskGenerator`.

The kmodels port provides:

| Function | What it corresponds to |
|---|---|
| `generate_crop_boxes` | `SamProcessor.generate_crop_boxes` |
| `filter_masks` | `SamProcessor.filter_masks` |
| `post_process_for_mask_generation` | `SamProcessor.post_process_for_mask_generation` |
| `SAMGenerateMasks` | `SamAutomaticMaskGenerator` (Meta original) |

All helpers run on `keras.ops` tensors and work on any backend.

### One-call usage

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kmodels.models.sam import SAMPromptableSegment, SAMGenerateMasks


def overlay_masks(ax, masks_list):
    rng = np.random.default_rng(7)
    ordered = sorted(
        [np.asarray(keras.ops.convert_to_numpy(m)).astype(bool) for m in masks_list],
        key=lambda m: -int(m.sum()),
    )
    h, w = ordered[0].shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for mask in ordered:
        color = np.concatenate([rng.random(3), [0.55]])
        overlay[mask] = color
    ax.imshow(overlay)


model = SAMPromptableSegment.from_weights("sam_vit_large")
img = Image.open("assets/coco_cats.jpg").convert("RGB")

result = SAMGenerateMasks(
    model,
    np.array(img, dtype="float32"),
    points_per_side=16,        # 16 x 16 = 256 grid points
    points_per_batch=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crops_nms_thresh=0.7,
    crop_n_layers=0,           # set 1 or 2 for multi-scale crops
)

# result["masks"]      : list of bool (orig_h, orig_w) keras tensors
# result["iou_scores"] : (N,) float tensor
# result["boxes"]      : (N, 4) xyxy in original-image coords
# result["rle_masks"]  : list of uncompressed RLE dicts
print(f"Found {len(result['masks'])} masks")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(np.array(img)); axes[0].set_title("Input"); axes[0].axis("off")
axes[1].imshow(np.array(img))
overlay_masks(axes[1], result["masks"])
axes[1].set_title(f"SAM ViT-Large - AMG ({len(result['masks'])} masks)")
axes[1].axis("off")
plt.tight_layout()
fig.savefig("sam_coco_cats_amg_output.jpg", bbox_inches="tight", dpi=130)
plt.close(fig)
```

![SAM Automatic Mask Generation Output](../assets/sam_coco_cats_amg_output.jpg)

Running on ``assets/coco_cats.jpg`` with a 16 × 16 point grid returns ~21 deduplicated masks — each cat as a whole, the two remote controls, the pink couch, and a handful of sub-parts (ears, paws, tail tips).

Under the hood the driver:
1. Calls `generate_crop_boxes` to build the point grid (and optional crop hierarchy).
2. Runs `model.get_image_embeddings` once per crop, then calls `model.prompt_decoder_model` in batches of `points_per_batch`.
3. Applies `filter_masks` per crop (IoU threshold, stability score, crop-edge filter, pad back to original image, encode as RLE).
4. Applies `post_process_for_mask_generation` (single-class NMS on the predicted boxes) to deduplicate across crops.

### Rolling your own driver

If you want HuggingFace-parity behavior exactly, import the helpers from the submodule and skip `SAMGenerateMasks`:

```python
from kmodels.models.sam.sam_image_processor import (
    generate_crop_boxes, filter_masks, post_process_for_mask_generation,
)
```

These map 1:1 to the HF equivalents, so you can mirror any custom pipeline written against `transformers`' `SamProcessor`. They are not part of the top-level public API — keep `SAMGenerateMasks` as the recommended entry point.

## Architecture

SAM consists of three main components:

1. **Vision Encoder (Image Encoder):** A ViT backbone with windowed attention and relative positional embeddings. Processes the input image (1024×1024) into a dense feature map (64×64×256).

2. **Prompt Encoder:** Encodes sparse prompts (points and box corners) via Fourier positional encoding + learned type embeddings, and dense prompts (masks) via a small CNN downsampling stack. A learned "no-mask" embedding is used when no mask prompt is supplied.

3. **Mask Decoder:** A lightweight two-way transformer (2 layers) that attends between prompt tokens and image embeddings, then generates mask predictions and IoU confidence scores via hypernetwork MLPs.

## Model Outputs

The model returns a dictionary with:
- `pred_masks`: Low-resolution predicted masks of shape `(batch, point_batch, num_masks, 256, 256)`, where `num_masks=3` for `multimask_output=True` (the default) or `num_masks=1` for `multimask_output=False`.
- `iou_scores`: Predicted IoU scores for each mask of shape `(batch, point_batch, num_masks)`.

Use `processor.post_process_masks(...)` to upscale masks to the original image resolution. The output is mask **logits** — threshold with `> 0` to get a binary mask (or whatever `mask_threshold` you prefer).

## HuggingFace API Parity Notes

The Keras port intentionally differs from the PyTorch/HuggingFace `SamModel` API in a few places due to the functional-graph constraint:

| Aspect | HuggingFace | Keras port |
|---|---|---|
| Optional prompts | pass `None` in `forward` | enable via `enable_boxes=True` / `enable_masks=True` at construction |
| `multimask_output` | runtime kwarg | construction-time flag |
| Precomputed embeddings | `model(image_embeddings=..., ...)` | `model.prompt_decoder_model(...)` sub-model |
| Post-processing | `processor.post_process_masks` (list per image) | `processor.post_process_masks` (one image per call) |
| Automatic mask generation | helpers only; driver lives in Meta's original repo | helpers + built-in `SAMGenerateMasks` driver |

All forward-pass weights are byte-equivalent to the HuggingFace checkpoints — the divergence is purely at the Python API surface.
