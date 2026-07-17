# SegFormer

**Paper**: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

SegFormer is a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perceptron (MLP) decoders. It comprises a hierarchically structured Transformer encoder which outputs multiscale features, and a lightweight All-MLP decoder which aggregates information from different layers.

Two classes are exposed:

- `SegFormerModel`: MiT hierarchical Transformer backbone (no decode head). Use as a feature extractor or to attach a custom head.
- `SegFormerSemanticSegment`: full semantic-segmentation model with the all-MLP decode head + classifier + bilinear upsample. This is what you instantiate to predict masks.

## Available Weights

Pretrained weights are loaded via `SegFormerSemanticSegment.from_weights(variant_id)` for kerasformers releases, or `SegFormerSemanticSegment.from_weights("hf:<repo>")` for arbitrary HF fine-tunes.

| Variant                          | Backbone | Dataset    | Classes | Input    |
|----------------------------------|----------|------------|--------:|----------|
| `segformer_b0_cityscapes_1024`   | MiT-B0   | Cityscapes |      19 | 1024×1024|
| `segformer_b0_cityscapes_768`    | MiT-B0   | Cityscapes |      19 | 768×768  |
| `segformer_b0_ade_512`           | MiT-B0   | ADE20K     |     150 | 512×512  |
| `segformer_b1_cityscapes_1024`   | MiT-B1   | Cityscapes |      19 | 1024×1024|
| `segformer_b1_ade_512`           | MiT-B1   | ADE20K     |     150 | 512×512  |
| `segformer_b2_cityscapes_1024`   | MiT-B2   | Cityscapes |      19 | 1024×1024|
| `segformer_b2_ade_512`           | MiT-B2   | ADE20K     |     150 | 512×512  |
| `segformer_b3_cityscapes_1024`   | MiT-B3   | Cityscapes |      19 | 1024×1024|
| `segformer_b3_ade_512`           | MiT-B3   | ADE20K     |     150 | 512×512  |
| `segformer_b4_cityscapes_1024`   | MiT-B4   | Cityscapes |      19 | 1024×1024|
| `segformer_b4_ade_512`           | MiT-B4   | ADE20K     |     150 | 512×512  |
| `segformer_b5_cityscapes_1024`   | MiT-B5   | Cityscapes |      19 | 1024×1024|
| `segformer_b5_ade_640`           | MiT-B5   | ADE20K     |     150 | 640×640  |

## Basic Usage

```python
from kerasformers.models.segformer import SegFormerSemanticSegment

model = SegFormerSemanticSegment.from_weights("segformer_b0_ade_512")
```

Build an untrained model (architecture only) for fine-tuning from scratch:

```python
model = SegFormerSemanticSegment.from_weights(
    "segformer_b0_ade_512", load_weights=False
)
```

Override any per-variant default (e.g. `num_classes` for fine-tuning):

```python
model = SegFormerSemanticSegment.from_weights(
    "segformer_b0_ade_512",
    load_weights=False,
    num_classes=10,
)
```

### Loading HF fine-tunes

Any HF repo whose `model_type` is `"segformer"` (the official NVIDIA checkpoints or arbitrary user fine-tunes) can be loaded directly via `from_weights("hf:<repo>")`. The class reads MiT dims, decoder dim, num classes, and image size straight from the HF config.

```python
model = SegFormerSemanticSegment.from_weights(
    "hf:nvidia/segformer-b0-finetuned-ade-512-512"
)
```

## Inference Example

```python
from kerasformers.models.segformer import SegFormerSemanticSegment, SegFormerImageProcessor

model = SegFormerSemanticSegment.from_weights("segformer_b0_ade_512")

processor = SegFormerImageProcessor(size={"height": 512, "width": 512})
inputs = processor("image.jpg")

output = model(inputs["pixel_values"], training=False)

result = processor.post_process_semantic_segmentation(output)
print(f"Detected classes: {result['class_names']}")

# Output:
# Detected classes: ['building', 'sky', 'tree', 'road', 'sidewalk',
#   'person', 'car', 'streetlight']
```

### Data format

Every processor and format-sensitive post-processor in this module accepts a `data_format=None` kwarg. The default (`None`) resolves to `keras.config.image_data_format()`; pass `"channels_first"` or `"channels_last"` to override per-call without touching global state.

```python
# follow the global config (the default)
processor = SegFormerImageProcessor()
inputs = processor("photo.jpg")

# force channels_first for this call only
processor = SegFormerImageProcessor(data_format="channels_first")
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

from kerasformers.models.segformer import SegFormerSemanticSegment, SegFormerImageProcessor

model = SegFormerSemanticSegment.from_weights("segformer_b0_ade_512")

img = Image.open("image.jpg").convert("RGB")
original_size = img.size[::-1]  # (H, W)

processor = SegFormerImageProcessor(size={"height": 512, "width": 512})
inputs = processor(img)
output = model(inputs["pixel_values"], training=False)

result = processor.post_process_semantic_segmentation(output, target_size=original_size)
mask_resized = result["segmentation"]

# Generate colors per class
np.random.seed(42)
colors = np.random.randint(50, 220, size=(150, 3), dtype=np.uint8)

colored_mask = colors[mask_resized % 150]
overlay = np.array(img).copy()
alpha = 0.55
overlay = (overlay * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.imshow(overlay)

# Legend for top classes by area
class_areas = [(c, (mask_resized == c).sum()) for c in result["unique_classes"]]
class_areas.sort(key=lambda x: -x[1])
top_classes = [c for c, _ in class_areas[:8]]
top_names = [n for c, n in zip(result["unique_classes"], result["class_names"]) if c in top_classes]

legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=colors[c % 150] / 255.0) for c in top_classes]
ax.legend(legend_patches, top_names, loc="upper right", fontsize=10)

ax.set_title("SegFormer Semantic Segmentation (ADE20K)", fontsize=16)
ax.axis("off")
plt.tight_layout()
fig.savefig("segformer_output.jpg", bbox_inches="tight", dpi=120)
plt.close(fig)
```

![SegFormer Semantic Segmentation Output](../assets/segformer_output.jpg)

## Custom Dataset Usage

When using a model fine-tuned on a custom dataset, pass your class names to the post-processor via `label_names`:

```python
# For any custom dataset
MY_CLASSES = ["background", "road", "building", "vegetation"]
result = processor.post_process_semantic_segmentation(output, target_size=original_size,
    label_names=MY_CLASSES)
```

If `label_names` is not provided, ADE20K class names (150 classes) are used by default.
