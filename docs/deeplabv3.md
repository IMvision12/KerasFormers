# DeepLabV3

**Paper**: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

DeepLabV3 is a highly accurate semantic segmentation model that employs atrous (dilated) convolution to capture multi-scale spatial context without losing spatial resolution. It features an Atrous Spatial Pyramid Pooling (ASPP) module that probes convolutional features at multiple scales, making it highly robust for segmenting objects of varying sizes.

Two classes are exposed:

- `DeepLabV3Model` — dilated ResNet backbone (no segmentation head). Returns the 2048-channel C5 feature at ``output_stride=8``.
- `DeepLabV3Segment` — full semantic-segmentation model with the ASPP module + classifier head + bilinear upsample.

## Architecture Highlights

- **ResNet Backbone:** Leverages deep residual networks (ResNet-50 or ResNet-101) for robust feature extraction.
- **Atrous Convolution:** Controls the resolution of features computed by Deep CNNs and effectively enlarges the field of view of filters without increasing the number of parameters or the amount of computation.
- **ASPP Module:** Captures multi-scale information by applying parallel atrous convolutions with different dilation rates (12, 24, 36).

## Available Weights

Pretrained weights are loaded via `DeepLabV3Segment.from_weights(variant_id)`. These come from torchvision (COCO + Pascal VOC fine-tune), not HuggingFace.

| Variant                           | Backbone   | Dataset       | Classes | Input    |
|-----------------------------------|------------|---------------|--------:|----------|
| `deeplabv3_resnet50_coco_voc`     | ResNet-50  | COCO + VOC    |      21 | 520×520  |
| `deeplabv3_resnet101_coco_voc`    | ResNet-101 | COCO + VOC    |      21 | 520×520  |

*Note: The `coco_voc` weights are pre-trained on the COCO dataset and fine-tuned on the PASCAL VOC segmentation dataset. They output predictions across 21 classes (20 objects + 1 background).*

## Basic Usage

```python
from kerasformers.models.deeplabv3 import DeepLabV3Segment

# Load model with pre-trained weights
model = DeepLabV3Segment.from_weights("deeplabv3_resnet50_coco_voc")

# Use the ResNet-101 backbone
model_large = DeepLabV3Segment.from_weights("deeplabv3_resnet101_coco_voc")

# Build an untrained model for fine-tuning (override num_classes etc.)
custom = DeepLabV3Segment.from_weights(
    "deeplabv3_resnet50_coco_voc",
    load_weights=False,
    num_classes=10,
    input_shape=(512, 512, 3),
)
```

## Inference Example

```python
from kerasformers.models.deeplabv3 import DeepLabV3Segment, DeepLabV3ImageProcessor

model = DeepLabV3Segment.from_weights("deeplabv3_resnet50_coco_voc")

processor = DeepLabV3ImageProcessor(size={"height": 520, "width": 520})
image = processor("image.jpg")

output = model(image["pixel_values"], training=False)  # (1, 520, 520, 21)

result = processor.post_process_semantic_segmentation(output)
print(f"Detected: {[c for c in result['class_names'] if c != 'background']}")

# Output:
# Detected: ['person']
```

### Data format

Every processor and format-sensitive post-processor in this module accepts a `data_format=None` kwarg. The default (`None`) resolves to `keras.config.image_data_format()`; pass `"channels_first"` or `"channels_last"` to override per-call without touching global state.

```python
# follow the global config (the default)
processor = DeepLabV3ImageProcessor()
inputs = processor("photo.jpg")

# force channels_first for this call only
processor = DeepLabV3ImageProcessor(data_format="channels_first")
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

from kerasformers.models.deeplabv3 import DeepLabV3Segment, DeepLabV3ImageProcessor

VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)

model = DeepLabV3Segment.from_weights("deeplabv3_resnet50_coco_voc")

img = Image.open("image.jpg").convert("RGB")
original_size = img.size[::-1]  # (H, W)

processor = DeepLabV3ImageProcessor(size={"height": 520, "width": 520})
inputs = processor(img)
output = model(inputs["pixel_values"], training=False)

result = processor.post_process_semantic_segmentation(output, target_size=original_size)
mask_resized = result["segmentation"]

colored_mask = VOC_COLORMAP[mask_resized]
overlay = np.array(img).copy()
alpha = 0.5
mask_pixels = mask_resized > 0
overlay[mask_pixels] = (overlay[mask_pixels] * (1 - alpha) + colored_mask[mask_pixels] * alpha).astype(np.uint8)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.imshow(overlay)

# Add legend
unique_classes = [c for c in result["unique_classes"] if c > 0]
legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=VOC_COLORMAP[c] / 255.0) for c in unique_classes]
if legend_patches:
    ax.legend(legend_patches, [n for c, n in zip(result["unique_classes"], result["class_names"]) if c > 0],
              loc="upper right", fontsize=11)

ax.set_title("DeepLabV3 Semantic Segmentation", fontsize=16)
ax.axis("off")
plt.tight_layout()
fig.savefig("deeplabv3_output.jpg", bbox_inches="tight", dpi=120)
plt.close(fig)
```

![DeepLabV3 Semantic Segmentation Output](../assets/deeplabv3_output.jpg)

## Custom Dataset Usage

When using a model fine-tuned on a custom dataset, pass your class names to the post-processor via `label_names`:

```python
MY_CLASSES = ["background", "crack", "pothole", "patch"]

result = processor.post_process_semantic_segmentation(output, target_size=original_size, label_names=MY_CLASSES)
```

If `label_names` is not provided, Pascal VOC class names (21 classes) are used by default.
