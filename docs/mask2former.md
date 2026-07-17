# Mask2Former

**Paper**: [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)

Mask2Former replaces MaskFormer's FPN pixel decoder with a 6-layer multi-scale deformable-attention (MSDeformAttn) encoder over the three coarsest backbone features and adds an FPN-style fusion step with the finest backbone stage to produce stride-4 mask features. The transformer decoder uses *masked* cross-attention: at each decoder layer, attention is restricted to positions where the previous layer's predicted mask is above 0.5, cycling through the three pixel-decoder scales across the 9 decoder layers.

Two classes are exposed:

- `Mask2FormerModel`: Swin backbone + MSDeformAttn pixel decoder + masked-attention transformer decoder + class/mask heads.
- `Mask2FormerUniversalSegment`: alias with the pretrained-weights registry attached.

## Architecture Highlights

- **MSDeformAttn pixel decoder:** 6 multi-scale deformable-attention encoder layers operate on the 3 coarsest backbone features (stages 2/3/4). Each query attends to 4 sampling points per head per level, with offsets and weights predicted from the query. A learned per-level embedding (`level_embed`) is added to the memory features.
- **FPN fusion:** The finest MSDeformAttn output (stride 8) is bilinearly upsampled and added to a lateral projection of the stride-4 backbone feature, refined with a 3×3 conv + GroupNorm + ReLU, then projected with a 1×1 conv to mask features.
- **Masked-attention decoder:** 9 decoder layers cycle through the 3 multi-scale features (one scale per layer, 3 passes). Cross-attention runs first (with the previous layer's predicted mask used as additive attention bias), then self-attention over queries, then FFN. The mask is repredicted after each layer and re-sized for the next layer's level.
- **Fused QKV cross-attention:** The cross-attention uses PyTorch's `nn.MultiheadAttention`-style fused `in_proj_weight: (3·d, d)` layout (sliced into Q/K/V at call time) so HF checkpoints transfer directly.
- **Iterative mask refinement:** Class and mask predictions are made *before each decoder layer* (using a shared layernorm + class predictor + 3-layer mask embedder MLP), and the final output is the prediction after the last layer.

## Available Weights

Pretrained weights are loaded via `Mask2FormerUniversalSegment.from_weights(variant_id)` for kerasformers releases, or `Mask2FormerUniversalSegment.from_weights("hf:<repo>")` for arbitrary HF fine-tunes.

| Variant | Backbone | Dataset | Classes | Queries | Input |
|---|---|---|---:|---:|---|
| `mask2former-swin-tiny-coco-instance` | Swin-Tiny | COCO instance | 80 | 100 | 384×384 |
| `mask2former-swin-small-coco-instance` | Swin-Small | COCO instance | 80 | 100 | 384×384 |
| `mask2former-swin-base-coco-instance` | Swin-Base | COCO instance | 80 | 100 | 384×384 |
| `mask2former-swin-large-coco-instance` | Swin-Large | COCO instance | 80 | 200 | 384×384 |
| `mask2former-swin-tiny-ade-semantic` | Swin-Tiny | ADE20K semantic | 150 | 100 | 512×512 |
| `mask2former-swin-tiny-coco-panoptic` | Swin-Tiny | COCO panoptic | 133 | 100 | 384×384 |

## Basic Usage

```python
from kerasformers.models.mask2former import Mask2FormerUniversalSegment

# COCO instance segmentation (Tiny)
model = Mask2FormerUniversalSegment.from_weights("mask2former-swin-tiny-coco-instance")

# Direct HF load
model = Mask2FormerUniversalSegment.from_weights(
    "hf:facebook/mask2former-swin-tiny-coco-instance"
)

# Build untrained architecture with a custom number of labels for fine-tuning
custom = Mask2FormerUniversalSegment.from_weights(
    "mask2former-swin-tiny-coco-instance",
    load_weights=False,
    num_labels=12,
)
```

## Inference Example

```python
from kerasformers.models.mask2former import (
    Mask2FormerUniversalSegment,
    Mask2FormerImageProcessor,
)
from PIL import Image

model = Mask2FormerUniversalSegment.from_weights("mask2former-swin-tiny-coco-instance")

image = Image.open("image.jpg").convert("RGB")

processor = Mask2FormerImageProcessor(target_size=384)
inputs = processor(image)

output = model(inputs["pixel_values"], training=False)
# output["class_queries_logits"]: (1, num_queries, num_labels + 1)
# output["masks_queries_logits"]:  (1, num_queries, H/4, W/4)
```

## Loading HF Fine-Tunes

Any HF repo whose `model_type` is `"mask2former"` loads directly via `from_weights("hf:<repo>")`. The class reads backbone config, hidden dim, decoder/encoder layers, and num_labels from the HF `config.json`.

```python
model = Mask2FormerUniversalSegment.from_weights(
    "hf:facebook/mask2former-swin-base-coco-panoptic"
)
```
