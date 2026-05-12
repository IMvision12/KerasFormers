# DINOv3

**Paper**: [DINOv3: Self-Supervised Visual Representation Learning at Scale](https://arxiv.org/abs/2508.10104)

DINOv3 is the third generation of the DINO self-supervised learning framework. It introduces 2D Rotary Position Embeddings (RoPE) and register tokens to the ViT backbone, and distills features into ConvNeXt-v2 student networks. Trained on the large-scale LVD-1689M dataset, DINOv3 produces state-of-the-art visual features for downstream tasks.

DINOv3 is a pure feature extractor — no classification head. Two backbone classes are exposed:

- `DinoV3ViTBackbone` — DINOv3 ViT with 2D RoPE + register tokens (3 variants).
- `DinoV3ConvNeXtBackbone` — DINOv3 ConvNeXt student (4 variants).

Both return the list of intermediate feature maps from each block / stage, suitable for feeding into detection / segmentation / depth necks.

## Weights License

DINOv3 weights are **gated** on HuggingFace and cannot be redistributed. The first call to `from_weights(variant)` downloads from HF, converts to Keras format, and caches the result at `~/.cache/kmodels/<variant>/`. Subsequent calls reload from cache.

Before the first call:

1. Accept the license at https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m (and the variant you want to load).
2. Authenticate with one of:
   - `huggingface-cli login`
   - `export HF_TOKEN=<your_token>`

## Available Weights

### `DinoV3ViTBackbone`

| Variant            | Backbone | Patch | Parameters |
|--------------------|----------|------:|-----------:|
| `dinov3_vits16`    | ViT-S/16 |    16 |     ~21 M  |
| `dinov3_vitb16`    | ViT-B/16 |    16 |     ~86 M  |
| `dinov3_vitl16`    | ViT-L/16 |    16 |    ~300 M  |

### `DinoV3ConvNeXtBackbone`

| Variant                  | Backbone     | Depths      | Parameters |
|--------------------------|--------------|-------------|-----------:|
| `dinov3_convnext_tiny`   | ConvNeXt-T   | [3,3,9,3]   |     ~29 M  |
| `dinov3_convnext_small`  | ConvNeXt-S   | [3,3,27,3]  |     ~50 M  |
| `dinov3_convnext_base`   | ConvNeXt-B   | [3,3,27,3]  |     ~89 M  |
| `dinov3_convnext_large`  | ConvNeXt-L   | [3,3,27,3]  |    ~198 M  |

## Features and Capabilities

- **Gated weight loading:** First call downloads from HuggingFace, converts from PyTorch, and caches at `~/.cache/kmodels/<variant>/`.
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
from kmodels.models.dino_v3 import DinoV3ViTBackbone, DinoV3ConvNeXtBackbone

# ViT — returns 13 intermediate feature maps (embed + 12 blocks)
model = DinoV3ViTBackbone.from_weights("dinov3_vits16")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(len(features), features[-1].shape)
# 13, (1, 201, 384)  — 1 CLS + 4 register + 196 patches

# ConvNeXt — returns 5 stage feature maps (stem + 4 stages)
model = DinoV3ConvNeXtBackbone.from_weights("dinov3_convnext_tiny")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(len(features), features[-1].shape)  # 5, (1, 7, 7, 768)
```

## Loading HF Fine-tunes

Any HF repo whose `model_type` is `"dinov3_vit"` or `"dinov3_convnext"` (the official DINOv3 checkpoints or any user fine-tune built on the same architectures) can be loaded directly via `from_weights("hf:<repo>")`:

```python
from kmodels.models.dino_v3 import DinoV3ViTBackbone

model = DinoV3ViTBackbone.from_weights("hf:facebook/dinov3-vitb16-pretrain-lvd1689m")
```

## Building a Classification Model on Top

```python
import keras
from kmodels.models.dino_v3 import DinoV3ViTBackbone

backbone = DinoV3ViTBackbone.from_weights("dinov3_vits16")
features = backbone.output  # list
cls_token = features[-1][:, 0]
logits = keras.layers.Dense(10, activation="softmax")(cls_token)
model = keras.Model(inputs=backbone.input, outputs=logits)

# Freeze the backbone, train only the head
backbone.trainable = False
```

## Cache Location

Converted weights live in `~/.cache/kmodels/<variant>/<variant>.weights.h5`. To force a re-download (e.g. after a converter update), delete the cache file.
