# MaskFormer

**Paper**: [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278)

MaskFormer reformulates per-pixel segmentation as a *mask classification* problem: a Swin backbone feeds an FPN-style pixel decoder that produces high-resolution mask features (stride 4), while a DETR-style transformer decoder runs `num_queries` learned object queries over the coarsest backbone feature to produce per-query class logits and mask embeddings. Final per-query masks are computed as a bilinear product of the mask embeddings and the pixel-decoder mask features.

Two classes are exposed:

- `MaskFormerModel`: Swin backbone + FPN pixel decoder + DETR-style transformer decoder + class/mask heads. Returns the segmentation output dict.
- `MaskFormerUniversalSegment`: alias with the pretrained-weights registry attached (use this for `from_weights("hf:…")` or release variants).

## Architecture Highlights

- **Mask classification framing:** A fixed set of `num_queries` queries each predict (class, mask) pairs, unifying semantic / instance / panoptic segmentation under one head.
- **Swin backbone with HF-aligned naming:** Separate `query`/`key`/`value` Dense projections and per-stage `hidden_states_norms` mirror HuggingFace's `MaskFormerSwinModel` so HF checkpoints transfer cleanly.
- **FPN pixel decoder:** Stem 3×3 conv on the coarsest stage, three lateral 1×1 + output 3×3 stages with `GroupNorm` (no bias on the conv), and a final 3×3 `mask_projection` to produce stride-4 mask features.
- **DETR-style transformer decoder:** Post-norm self-attention + cross-attention + FFN with sinusoidal 2-D position embedding on the encoder memory and learned object-query positional embedding.

## Available Weights

Pretrained weights are loaded via `MaskFormerUniversalSegment.from_weights(variant_id)` for kerasformers releases, or `MaskFormerUniversalSegment.from_weights("hf:<repo>")` for arbitrary HF fine-tunes.

| Variant | Backbone | Dataset | Classes | Queries | Input |
|---|---|---|---:|---:|---|
| `maskformer-swin-tiny-ade` | Swin-Tiny | ADE20K semantic | 150 | 100 | 512×512 |
| `maskformer-swin-tiny-coco` | Swin-Tiny | COCO panoptic | 133 | 100 | 384×384 |
| `maskformer-swin-small-coco` | Swin-Small | COCO panoptic | 133 | 100 | 384×384 |
| `maskformer-swin-base-ade` | Swin-Base | ADE20K semantic | 150 | 100 | 512×512 |
| `maskformer-swin-base-coco` | Swin-Base | COCO panoptic | 133 | 100 | 384×384 |

## Basic Usage

```python
from kerasformers.models.maskformer import MaskFormerUniversalSegment

# ADE20K semantic (Tiny)
model = MaskFormerUniversalSegment.from_weights("maskformer-swin-tiny-ade")

# Direct HF load
model = MaskFormerUniversalSegment.from_weights("hf:facebook/maskformer-swin-tiny-coco")

# Build untrained architecture with a custom number of labels for fine-tuning
custom = MaskFormerUniversalSegment.from_weights(
    "maskformer-swin-tiny-ade",
    load_weights=False,
    num_labels=12,
)
```

## Inference Example

```python
from kerasformers.models.maskformer import MaskFormerUniversalSegment, MaskFormerImageProcessor
from PIL import Image

model = MaskFormerUniversalSegment.from_weights("maskformer-swin-tiny-ade")

image = Image.open("image.jpg").convert("RGB")
original_h, original_w = image.size[1], image.size[0]

processor = MaskFormerImageProcessor(target_size=512)
inputs = processor(image)

output = model(inputs["pixel_values"], training=False)
# output["class_queries_logits"]: (1, num_queries, num_labels + 1)
# output["masks_queries_logits"]:  (1, num_queries, H/4, W/4)

semantic = processor.post_process_semantic_segmentation(
    output, target_sizes=[(original_h, original_w)],
)
# semantic[0]: (H, W) int32, per-pixel class id
```

## Segmentation Modes

`MaskFormerImageProcessor` exposes the segmentation post-processors:

```python
processor = MaskFormerImageProcessor(target_size=512)
output = model(processor(image)["pixel_values"], training=False)

# Semantic (per-pixel class id, no instance separation)
semantic = processor.post_process_semantic_segmentation(
    output, target_sizes=[(image.height, image.width)],
)

# Panoptic (things + stuff merged into one segmentation)
panoptic = processor.post_process_panoptic_segmentation(
    output, target_size=(image.height, image.width), threshold=0.8,
)
```

| Method | Returns |
|---|---|
| `post_process_semantic_segmentation` | List of `(H, W)` int32 arrays (one per image) |
| `post_process_panoptic_segmentation` | `{"segmentation": (H, W) int32, "segments_info": [...]}` |

## Loading HF Fine-Tunes

Any HF repo whose `model_type` is `"maskformer"` (the official `facebook/...` checkpoints or arbitrary user fine-tunes) loads directly via `from_weights("hf:<repo>")`. The class reads backbone config, decoder dims, and num_labels from the HF `config.json`.

```python
model = MaskFormerUniversalSegment.from_weights("hf:facebook/maskformer-swin-base-coco")
```
