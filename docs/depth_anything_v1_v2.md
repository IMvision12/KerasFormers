# Depth Anything V1 & V2

**V1 Paper**: [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891)
**V2 Paper**: [Depth Anything V2](https://arxiv.org/abs/2406.09414)

Depth Anything is a monocular depth-estimation model family that pairs a
DINOv2 ViT backbone with a DPT-style neck and head. V1 trains on a mix of
labeled images and very large-scale pseudo-labeled images; V2 keeps the same
architecture but replaces the labeled real images with synthetic data and
scales up the teacher model to produce noticeably sharper and more robust
depth maps. The V2 release also ships metric-depth variants fine-tuned for
indoor and outdoor scenes.

Both versions share the same Keras implementation. Two classes are exposed
per version:

- `DepthAnythingV{1,2}Model` — backbone + DPT neck only (no head). Use as a
  feature extractor or to attach a custom head.
- `DepthAnythingV{1,2}DepthEstimation` — full monocular depth estimator
  with the depth head. This is what you instantiate to predict depth.

## Architecture

1. **DINOv2 backbone** (`depth_anything_v1_dino_backbone`) — patch embed +
   CLS token + position embeddings + `backbone_depth` pre-norm transformer
   blocks with LayerScale on both branches. Returns four intermediate
   feature maps at the block indices listed in `out_indices`.
2. **DPT neck** (`depth_anything_v1_neck`) — reassemble (1x1 projection +
   per-factor up/down sampling), project to `fusion_hidden_size` with 3x3
   convs, and walk the pyramid bottom-up through four fusion stages.
3. **Depth head** (`depth_anything_v1_head`) — three convs with an
   aligned-corners bilinear upsample to the input resolution between the
   first and second conv. Relative variants end in a `ReLU`, metric
   variants end in a `sigmoid` scaled by `max_depth`.

The fusion and head upsamples use a pure-Keras
`depth_anything_v1_aligned_bilinear_resize` that matches
`torch.nn.functional.interpolate(..., align_corners=True)` via explicit
gather + lerp, so the model is numerically consistent across `torch`,
`jax`, and `tensorflow` backends and respects
`keras.config.image_data_format()` end-to-end.

## Available Weights

Pretrained weights are loaded via `from_weights(variant_id)` (or
`from_hf(hf_id)` for arbitrary HF fine-tunes).

### Relative Depth

| Variant                   | Class                            | Parameters | Backbone        |
|---------------------------|----------------------------------|-----------:|-----------------|
| `depth_anything_small`    | `DepthAnythingV1DepthEstimation` |     ~24 M  | DINOv2 ViT-S/14 |
| `depth_anything_base`     | `DepthAnythingV1DepthEstimation` |     ~97 M  | DINOv2 ViT-B/14 |
| `depth_anything_large`    | `DepthAnythingV1DepthEstimation` |    ~335 M  | DINOv2 ViT-L/14 |
| `depth_anything_v2_small` | `DepthAnythingV2DepthEstimation` |     ~24 M  | DINOv2 ViT-S/14 |
| `depth_anything_v2_base`  | `DepthAnythingV2DepthEstimation` |     ~97 M  | DINOv2 ViT-B/14 |
| `depth_anything_v2_large` | `DepthAnythingV2DepthEstimation` |    ~335 M  | DINOv2 ViT-L/14 |

### Metric Depth (V2 only)

| Variant                                         | Max depth | Description                            |
|-------------------------------------------------|----------:|----------------------------------------|
| `depth_anything_v2_metric_indoor_small`         |    20 m   | Indoor metric depth (NYUv2 fine-tuned) |
| `depth_anything_v2_metric_indoor_base`          |    20 m   | Indoor metric depth                    |
| `depth_anything_v2_metric_indoor_large`         |    20 m   | Indoor metric depth                    |
| `depth_anything_v2_metric_outdoor_small`        |    80 m   | Outdoor metric depth (KITTI-style)     |
| `depth_anything_v2_metric_outdoor_base`         |    80 m   | Outdoor metric depth                   |
| `depth_anything_v2_metric_outdoor_large`        |    80 m   | Outdoor metric depth                   |

All variants default to a 518×518 input (37x37 DINOv2 patch grid).

## Image Processor

Both `kmodels.models.depth_anything_v1` and
`kmodels.models.depth_anything_v2` ship a pure-Keras image processor that
resizes an input image with bicubic interpolation, rescales to `[0, 1]`,
and applies ImageNet normalization. Unlike HF `DPTImageProcessor` — which
preserves the aspect ratio and produces a variable-shape output — this
processor stretches the image directly to the target size so the shape
matches what the Keras model was built with.

- `DepthAnythingV1ImageProcessor(target_size=518)(image)` / `DepthAnythingV2ImageProcessor()(...)`
- `processor.post_process_depth_estimation(predicted_depth, original_size)` (method on either processor)

`target_size` accepts either a single `int` (square output) or a
`(height, width)` tuple. Both dimensions should be multiples of the
DINOv2 patch size (14). The pretrained 518×518 position embeddings are
bilinearly interpolated to the new grid when weights are loaded, so
non-518 inputs work as long as the model was built with the same shape.

```python
from kmodels.models.depth_anything_v1 import (
    DepthAnythingV1DepthEstimation,
    DepthAnythingV1ImageProcessor,
)

model = DepthAnythingV1DepthEstimation.from_weights("depth_anything_small")
processor = DepthAnythingV1ImageProcessor()
inputs = processor("photo.jpg")
depth = model(inputs["pixel_values"])
depth_full = processor.post_process_depth_estimation(
    depth, original_size=inputs["original_size"]
)
print(depth_full.shape)  # (1, orig_h, orig_w)
```

## Basic Usage

### Relative Depth with V1

End-to-end example that loads an image, runs the small V1 model, and
saves a side-by-side RGB + depth visualization:

```python
import keras
import numpy as np
from PIL import Image
import matplotlib.cm as cm

from kmodels.models.depth_anything_v1 import (
    DepthAnythingV1DepthEstimation,
    DepthAnythingV1ImageProcessor,
)

# 1) build model + load pretrained weights
model = DepthAnythingV1DepthEstimation.from_weights("depth_anything_small")

# 2) preprocess the image (stretches to 518x518, ImageNet-normalized)
processor = DepthAnythingV1ImageProcessor()
inputs = processor("assets/coco_horse_dog.jpg")
orig_h, orig_w = inputs["original_size"]

# 3) forward pass — raw depth at model resolution
raw_depth = model(inputs["pixel_values"], training=False)

# 4) resample depth back to the original image size
depth = processor.post_process_depth_estimation(
    raw_depth, original_size=(orig_h, orig_w)
)
depth = keras.ops.convert_to_numpy(depth)[0]   # (orig_h, orig_w) float32

# 5) visualize: normalize + apply inferno colormap, save side-by-side
dn = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-8)
depth_color = (cm.inferno(dn)[..., :3] * 255).astype(np.uint8)
rgb = np.array(Image.open("assets/coco_horse_dog.jpg").convert("RGB").resize((orig_w, orig_h)))
side = np.concatenate([rgb, depth_color], axis=1)
Image.fromarray(side).save("depth_output.png")
```

Output (horse + dog in snow — closer objects are brighter):

![DepthAnythingV1 output](../assets/depth_anything_v1_output.jpg)

### Relative Depth with V2

Same API as V1 — swap the module and the variant name. V2 uses the same
processor / post-processor contract, just with sharper and more robust
depth thanks to its synthetic-data training set.

```python
import keras
import numpy as np
from PIL import Image
import matplotlib.cm as cm

from kmodels.models.depth_anything_v2 import (
    DepthAnythingV2DepthEstimation,
    DepthAnythingV2ImageProcessor,
)

model = DepthAnythingV2DepthEstimation.from_weights("depth_anything_v2_base")
processor = DepthAnythingV2ImageProcessor()
inputs = processor("assets/valley.png")
orig_h, orig_w = inputs["original_size"]

raw_depth = model(inputs["pixel_values"], training=False)
depth = processor.post_process_depth_estimation(
    raw_depth, original_size=(orig_h, orig_w)
)
depth = keras.ops.convert_to_numpy(depth)[0]

dn = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-8)
depth_color = (cm.inferno(dn)[..., :3] * 255).astype(np.uint8)
rgb = np.array(Image.open("assets/valley.png").convert("RGB").resize((orig_w, orig_h)))
side = np.concatenate([rgb, depth_color], axis=1)
Image.fromarray(side).save("depth_output.png")
```

Output (mountain valley — crisp ridges and foreground detail):

![DepthAnythingV2 output](../assets/depth_anything_v2_output.jpg)

### Metric Indoor Depth (V2)

```python
from kmodels.models.depth_anything_v2 import (
    DepthAnythingV2DepthEstimation,
    DepthAnythingV2ImageProcessor,
)

model = DepthAnythingV2DepthEstimation.from_weights(
    "depth_anything_v2_metric_indoor_large"
)
processor = DepthAnythingV2ImageProcessor()
inputs = processor("room.jpg")
depth = model(inputs["pixel_values"])
depth_full = processor.post_process_depth_estimation(
    depth, original_size=inputs["original_size"]
)
# depth_full values are in meters, bounded to [0, 20]
```

### Metric Outdoor Depth (V2)

```python
from kmodels.models.depth_anything_v2 import DepthAnythingV2DepthEstimation

model = DepthAnythingV2DepthEstimation.from_weights(
    "depth_anything_v2_metric_outdoor_large"
)
# ... same processor + post-process flow, depth bounded to [0, 80]
```

### Loading HF fine-tunes

Any HF repo whose `model_type` is `"depth_anything"` (the official V1/V2
checkpoints, the metric variants, or arbitrary user fine-tunes built on
those architectures) can be loaded directly with `from_hf`. The class
reads backbone dims, neck/fusion sizes, reassemble factors,
`depth_estimation_type`, and `max_depth` straight from the HF config.

```python
from kmodels.models.depth_anything_v2 import DepthAnythingV2DepthEstimation

model = DepthAnythingV2DepthEstimation.from_hf(
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    input_shape=(518, 518, 3),
)
```

## Non-518 Input Shapes

Both versions accept any input shape whose height and width are
multiples of 14. The DINOv2 position embeddings are resampled to the new
patch grid when the pretrained weights are loaded (via
`AddPositionEmbs.load_own_variables`), and the fusion-block upsample
targets are derived from the model's construction-time `input_shape`, so
each instance is locked to the shape you pick at build time.

```python
from kmodels.models.depth_anything_v2 import (
    DepthAnythingV2DepthEstimation,
    DepthAnythingV2ImageProcessor,
)

# Non-square 392x784 (28x56 patch grid) with pretrained weights
model = DepthAnythingV2DepthEstimation.from_weights(
    "depth_anything_v2_small",
    input_shape=(392, 784, 3),
)
processor = DepthAnythingV2ImageProcessor(target_size=(392, 784))
inputs = processor("photo.jpg")
depth = model(inputs["pixel_values"])
```

## Channels-First vs Channels-Last

Both versions follow `keras.config.image_data_format()` end-to-end. The
backbone patch embed, neck convs, head convs, and the aligned-corners
bilinear upsample all dispatch on the global data format, so switching
between `channels_last` and `channels_first` requires no manual
transposes.

```python
import keras
keras.config.set_image_data_format("channels_first")

from kmodels.models.depth_anything_v1 import DepthAnythingV1DepthEstimation
model = DepthAnythingV1DepthEstimation.from_weights("depth_anything_small")
# model input: (B, 3, 518, 518)  /  output: (B, 1, 518, 518)
```

Weight conversion always runs in `channels_last` + torch backend.

### Data format

Every processor and format-sensitive post-processor in this module accepts a `data_format=None` kwarg. The default (`None`) resolves to `keras.config.image_data_format()`; pass `"channels_first"` or `"channels_last"` to override per-call without touching global state.

```python
# follow the global config (the default)
processor = DepthAnythingV1ImageProcessor()
inputs = processor("photo.jpg")

# force channels_first for this call only
processor = DepthAnythingV1ImageProcessor(data_format="channels_first")
inputs = processor("photo.jpg")
```

Image processors return tensors in the requested layout; post-processors accept tensors in either layout and read the flag to pick the channel axis. See `docs/utils.md` for which families have format-sensitive post-processors.

## Model Outputs

- **Relative variants** return non-negative disparity-style depth. Useful
  for depth ordering, monocular SLAM initialization, and cases where only
  relative depth is needed.
- **Metric indoor variants** return metric depth in meters bounded to
  `[0, 20]`.
- **Metric outdoor variants** return metric depth in meters bounded to
  `[0, 80]`.

Output shape is `(batch, height, width, 1)` in `channels_last` or
`(batch, 1, height, width)` in `channels_first`. Use
`processor.post_process_depth_estimation(...)` to resample back to the
original image size and squeeze the channel dimension.

## Citations

```bibtex
@inproceedings{yang2024depth,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```

```bibtex
@article{yang2024depthv2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv preprint arXiv:2406.09414},
  year={2024}
}
```
