# MobileViT / MobileViTV2

**Papers:**

- MobileViT: [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)
- MobileViTV2: [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680)

MobileViT is a hybrid CNN-Transformer backbone designed for mobile inference. It interleaves MobileNetV2-style inverted-residual (MBConv) blocks with **MobileViT blocks** that fold local self-attention over fixed-size patches, mixing convolutional locality with transformer global context. MobileViTV2 replaces the quadratic multi-head attention with a **separable (linear) self-attention** that scales linearly in the number of patches, making it more efficient on mobile hardware while keeping the same 5-stage hierarchical layout.

Both versions ship as four classes:

- `MobileViTModel` / `MobileViTV2Model` — backbone (no head). Returns the last-stage feature map at `output_stride=32`.
- `MobileViTImageClassify` / `MobileViTV2ImageClassify` — full ImageNet classifier (backbone + 1×1 head conv + GAP + Dense).
- `MobileViTSegment` / `MobileViTV2Segment` — full DeepLabV3 semantic segmenter (backbone with `output_stride=16` and atrous convolutions in the last stage + ASPP module + 1×1 classifier conv).
- `MobileViTImageProcessor` / `MobileViTV2ImageProcessor` — matching preprocessor: resize-shortest-edge + center-crop + rescale + RGB→BGR flip.

## Architecture Highlights

- **Hybrid CNN-Transformer stages.** Stages 0–1 are pure MBConv; stages 2–4 are MBConv + MobileViT block (3×3 local conv → 1×1 expand → unfold-to-patches → transformer encoder layers → fold-back → 1×1 project → fuse).
- **Linear (separable) self-attention in V2.** The standard quadratic MHSA is replaced by a single `Conv1x1(C → 2C+1)` that produces fused `[query | key | value]`, then a softmax-over-patches followed by a context multiply and a `Conv1x1(C → C)` output projection. No layer-norm; uses GroupNorm(1, …) instead.
- **Atrous (dilated) convolutions for segmentation.** When `output_stride=16` (the DeepLabV3 fine-tune setting), stage 4 uses stride 1 instead of 2 and dilation 2 in the MV block's 3×3 conv so the feature map stays at H/16 while keeping the same receptive field as the classification path.
- **BGR channel order.** Both V1 and V2 were trained with channel-flipped input. The matching image processor flips RGB→BGR by default (`do_flip_channel_order=True`).

## Available HF Repos

All HuggingFace `model_type == "mobilevit"` or `"mobilevitv2"` repos load via `from_weights("hf:user/repo")`. The official Apple checkpoints:

| Class | HF Repo | Task | Classes | Input |
|---|---|---|---|---|
| `MobileViTImageClassify`   | `apple/mobilevit-xx-small`              | ImageNet classify     | 1000 | 256×256 |
| `MobileViTImageClassify`   | `apple/mobilevit-x-small`               | ImageNet classify     | 1000 | 256×256 |
| `MobileViTImageClassify`   | `apple/mobilevit-small`                 | ImageNet classify     | 1000 | 256×256 |
| `MobileViTSegment`         | `apple/deeplabv3-mobilevit-xx-small`    | PASCAL VOC segment    |   21 | 512×512 |
| `MobileViTSegment`         | `apple/deeplabv3-mobilevit-x-small`     | PASCAL VOC segment    |   21 | 512×512 |
| `MobileViTSegment`         | `apple/deeplabv3-mobilevit-small`       | PASCAL VOC segment    |   21 | 512×512 |
| `MobileViTV2ImageClassify` | `apple/mobilevitv2-{0.5,…,2.0}-imagenet1k-256` | ImageNet classify | 1000 | 256×256 |
| `MobileViTV2Segment`       | `apple/mobilevitv2-1.0-voc-deeplabv3`   | PASCAL VOC segment    |   21 | 512×512 |

Community fine-tunes work the same way — the model auto-detects `num_classes` from the HF config's `num_labels` (or falls back to `len(id2label)`).

## Basic Usage

```python
from kerasformers.models.mobilevit import (
    MobileViTImageClassify, MobileViTImageProcessor, MobileViTSegment,
)
from kerasformers.models.mobilevitv2 import (
    MobileViTV2ImageClassify, MobileViTV2ImageProcessor, MobileViTV2Segment,
)

# V1 classifier
model = MobileViTImageClassify.from_weights("hf:apple/mobilevit-small")

# V1 segmenter
seg = MobileViTSegment.from_weights("hf:apple/deeplabv3-mobilevit-small")

# V2 classifier
model_v2 = MobileViTV2ImageClassify.from_weights(
    "hf:apple/mobilevitv2-1.0-imagenet1k-256"
)

# V2 segmenter
seg_v2 = MobileViTV2Segment.from_weights("hf:apple/mobilevitv2-1.0-voc-deeplabv3")

# Community fine-tune (any HF repo with model_type=="mobilevit" works)
custom = MobileViTImageClassify.from_weights(
    "hf:annasabdurrahman354/mobilevit-xx-small-finetuned-eurosat"
)
print(custom.num_classes)  # auto-detected from id2label
```

## Classification Inference

```python
from kerasformers.models.mobilevit import (
    MobileViTImageClassify, MobileViTImageProcessor,
)

model = MobileViTImageClassify.from_weights(
    "hf:apple/mobilevit-small", include_normalization=False
)
processor = MobileViTImageProcessor()  # defaults: shortest_edge=288, crop=256

inputs = processor("image.jpg")
logits = model(inputs["pixel_values"], training=False)
pred_class = int(keras.ops.argmax(logits, axis=-1)[0])
```

`include_normalization=False` is important when feeding tensors from the processor — the processor already handles MobileViT's pixel scaling, so the in-model `ImageNormalizationLayer` would double-normalize. Either build the model with `include_normalization=False` and use the processor, or build with `include_normalization=True` (default) and pass raw images.

## Segmentation Inference

```python
from kerasformers.models.mobilevit import (
    MobileViTSegment, MobileViTImageProcessor,
)

model = MobileViTSegment.from_weights(
    "hf:apple/deeplabv3-mobilevit-small", include_normalization=False
)
processor = MobileViTImageProcessor(
    size={"shortest_edge": 544},
    crop_size={"height": 512, "width": 512},
)

inputs = processor("image.jpg")
logits = model(inputs["pixel_values"], training=False)  # (1, 32, 32, 21)

# Argmax + bilinear upsample to original image size
result = processor.post_process_semantic_segmentation(
    logits, target_size=(orig_h, orig_w)
)
print(result["class_names"])  # ['background', 'person']
```

Segmentation logits are returned at the feature resolution (`input_size / output_stride`), matching HF's behavior. The post-processor handles the argmax + nearest-neighbour resize back to the original image size.

## Image Processor

```python
MobileViTImageProcessor(
    size={"shortest_edge": 288},
    crop_size={"height": 256, "width": 256},
    resample="bilinear",
    do_resize=True,
    do_center_crop=True,
    do_rescale=True,
    rescale_factor=1 / 255,
    do_flip_channel_order=True,
    return_tensor=True,
    data_format=None,
)
```

Pipeline (matches HuggingFace `MobileViTImageProcessor`):

1. Resize so the shortest edge equals `size["shortest_edge"]` (preserves aspect ratio).
2. Center-crop to `crop_size["height"] × crop_size["width"]`.
3. Rescale pixel values by `rescale_factor` (default `1/255`).
4. Flip channel order RGB → BGR (`do_flip_channel_order=True`) — required to match how MobileViT was trained.
5. Return `{"pixel_values": tensor}` in the configured `data_format`.

No mean/std normalization — that's the HF default for MobileViT, and the matching keras checkpoints are trained the same way.

## Fine-tuning with a Different Class Count

For both classification and segmentation, point `from_weights` at any HF repo and the model auto-builds with the right head size:

```python
# 10-class fine-tune of EuroSAT, segmentation with custom class count, etc.
model = MobileViTImageClassify.from_weights("hf:user/my-mobilevit-finetune")
```

To train your own head from scratch on a kerasformers-released backbone:

```python
backbone = MobileViTModel.from_weights("mobilevit_s_cvnets_in1k")
classifier = keras.Sequential([
    backbone,
    keras.layers.Conv2D(640, 1, use_bias=False),
    keras.layers.BatchNormalization(epsilon=1e-5),
    keras.layers.Activation("swish"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
```

## Data Format

`MobileViTImageProcessor` accepts `data_format=None`. The default resolves to `keras.config.image_data_format()`; pass `"channels_first"` or `"channels_last"` to override per-call without touching global state. The matching `post_process_semantic_segmentation` accepts the same flag and reads the channel axis accordingly.
