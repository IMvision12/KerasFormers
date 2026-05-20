# DINOv3

**Paper**: [DINOv3: Self-Supervised Visual Representation Learning at Scale](https://arxiv.org/abs/2508.10104)

DINOv3 is the third generation of the DINO self-supervised learning framework. It introduces 2D Rotary Position Embeddings (RoPE) and register tokens to the ViT backbone, and distills features into ConvNeXt-v2 student networks. Trained on the large-scale LVD-1689M dataset, DINOv3 produces state-of-the-art visual features for downstream tasks.

DINOv3 is a pure feature extractor — no classification head. Two model classes are exposed:

- `DinoV3ViTModel` — DINOv3 ViT with 2D RoPE + register tokens (3 variants).
- `DinoV3ConvNeXtModel` — DINOv3 ConvNeXt student (4 variants).

Both accept an `as_backbone` flag (like the classification models): with `as_backbone=True` they return the list of intermediate feature maps from each block / stage (suitable for detection / segmentation / depth necks); with `as_backbone=False` (default) they return only the final output (ViT: the LN-normalized token sequence; ConvNeXt: the final-stage feature map).

## Weights License

DINOv3 weights are **gated** on HuggingFace and cannot be redistributed. The first call to `from_weights(variant)` downloads from HF, converts to Keras format, and caches the result at `~/.cache/kerasformers/<variant>/`. Subsequent calls reload from cache.

Before the first call:

1. Accept the license at https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m (and the variant you want to load).
2. Authenticate with one of:
   - `huggingface-cli login`
   - `export HF_TOKEN=<your_token>`

## Available Weights

### `DinoV3ViTModel`

| Variant            | Backbone | Patch | Parameters |
|--------------------|----------|------:|-----------:|
| `dinov3_vits16`    | ViT-S/16 |    16 |     ~21 M  |
| `dinov3_vitb16`    | ViT-B/16 |    16 |     ~86 M  |
| `dinov3_vitl16`    | ViT-L/16 |    16 |    ~300 M  |

### `DinoV3ConvNeXtModel`

| Variant                  | Backbone     | Depths      | Parameters |
|--------------------------|--------------|-------------|-----------:|
| `dinov3_convnext_tiny`   | ConvNeXt-T   | [3,3,9,3]   |     ~29 M  |
| `dinov3_convnext_small`  | ConvNeXt-S   | [3,3,27,3]  |     ~50 M  |
| `dinov3_convnext_base`   | ConvNeXt-B   | [3,3,27,3]  |     ~89 M  |
| `dinov3_convnext_large`  | ConvNeXt-L   | [3,3,27,3]  |    ~198 M  |

## Features and Capabilities

- **Gated weight loading:** First call downloads from HuggingFace, converts from PyTorch, and caches at `~/.cache/kerasformers/<variant>/`.
- **2D RoPE:** ViT variants use 2D Rotary Position Embeddings applied to patch tokens only (CLS and register tokens are excluded).
- **Register tokens:** 4 learnable register tokens inserted between CLS and patch tokens improve attention map quality.
- **HF passthrough:** `from_weights("hf:facebook/dinov3-...")` also works for arbitrary fine-tunes whose `model_type` is `"dinov3_vit"` or `"dinov3_convnext"`.
- **Fine-tune compatibility:** `query_bias`, `key_bias`, `value_bias`, `hidden_act`, `mlp_bias`, and `layer_norm_eps` are read from the HF config — fine-tunes that change these from the canonical DINOv3 settings (including gated-MLP variants) load correctly.

## Basic Usage

```python
import os
# Either login via CLI:
#   huggingface-cli login
# Or export the token (one-time):
os.environ["HF_TOKEN"] = "your_huggingface_token"

import numpy as np
from kerasformers.models.dino_v3 import DinoV3ViTModel, DinoV3ConvNeXtModel

# ViT, as a backbone — returns 13 intermediate feature maps (embed + 12 blocks)
model = DinoV3ViTModel.from_weights("dinov3_vits16", as_backbone=True)
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(len(features), features[-1].shape)
# 13, (1, 201, 384)  — 1 CLS + 4 register + 196 patches

# Default (as_backbone=False) — returns only the final LN-normalized token sequence
model = DinoV3ViTModel.from_weights("dinov3_vits16")
out = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(out.shape)  # (1, 201, 384)

# ConvNeXt, as a backbone — returns 5 stage feature maps (stem + 4 stages)
model = DinoV3ConvNeXtModel.from_weights("dinov3_convnext_tiny", as_backbone=True)
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(len(features), features[-1].shape)  # 5, (1, 7, 7, 768)
```

## Loading HF Fine-tunes

Any HF repo whose `model_type` is `"dinov3_vit"` or `"dinov3_convnext"` (the official DINOv3 checkpoints or any user fine-tune built on the same architectures) can be loaded directly via `from_weights("hf:<repo>")`:

```python
from kerasformers.models.dino_v3 import DinoV3ViTModel

model = DinoV3ViTModel.from_weights("hf:facebook/dinov3-vitb16-pretrain-lvd1689m")
```

## Building a Classification Model on Top

```python
import keras
from kerasformers.models.dino_v3 import DinoV3ViTModel

backbone = DinoV3ViTModel.from_weights("dinov3_vits16")
# Default output is the final LN-normalized token sequence; take the CLS token
cls_token = backbone.output[:, 0]
logits = keras.layers.Dense(10, activation="softmax")(cls_token)
model = keras.Model(inputs=backbone.input, outputs=logits)

# Freeze the backbone, train only the head
backbone.trainable = False
```

## Cache Location

Converted weights live in `~/.cache/kerasformers/<variant>/<variant>.weights.h5`. To force a re-download (e.g. after a converter update), delete the cache file.
