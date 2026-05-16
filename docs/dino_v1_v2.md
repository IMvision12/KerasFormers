# DINO & DINOv2

**DINO Paper**: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
**DINOv2 Paper**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

DINO (self-**DI**stillation with **NO** labels) is a self-supervised learning method for Vision Transformers. It produces rich visual features without any labeled data, making the models excellent general-purpose feature extractors for downstream tasks like segmentation, detection, retrieval, and depth estimation.

DINOv2 improves on DINO with a larger curated dataset (LVD-142M), LayerScale, and stronger training recipe, producing state-of-the-art visual features.

DINO and DINOv2 are pure feature extractors — no classification head. Three backbone classes are exposed:

- `DinoViTBackbone` — DINO V1 ViT (4 variants).
- `DinoResNetBackbone` — DINO V1 ResNet-50.
- `DinoV2Backbone` — DINOv2 ViT (3 variants). Supports `from_weights("hf:<repo>")` for HF Hub fine-tunes.

All three return a list of intermediate feature maps from each block / stage, suitable for feeding into detection / segmentation / depth necks.

## Available Weights

Pretrained weights are loaded via `Cls.from_weights(variant_id)`. `DinoV2Backbone` also supports `from_weights("hf:<repo>")` for arbitrary HF fine-tunes whose `model_type` is `"dinov2"`.

### DINO V1 (`DinoViTBackbone` and `DinoResNetBackbone`)

| Variant            | Class                | Backbone   | Patch | Parameters |
|--------------------|----------------------|------------|------:|-----------:|
| `dino_vits16`      | `DinoViTBackbone`    | ViT-S/16   |    16 |     ~21 M  |
| `dino_vits8`       | `DinoViTBackbone`    | ViT-S/8    |     8 |     ~21 M  |
| `dino_vitb16`      | `DinoViTBackbone`    | ViT-B/16   |    16 |     ~85 M  |
| `dino_vitb8`       | `DinoViTBackbone`    | ViT-B/8    |     8 |     ~85 M  |
| `dino_resnet50`    | `DinoResNetBackbone` | ResNet-50  |     — |     ~23 M  |

### DINOv2 (`DinoV2Backbone`)

| Variant           | Backbone   | Patch | Parameters |
|-------------------|------------|------:|-----------:|
| `dinov2_vits14`   | ViT-S/14   |    14 |     ~22 M  |
| `dinov2_vitb14`   | ViT-B/14   |    14 |     ~86 M  |
| `dinov2_vitl14`   | ViT-L/14   |    14 |    ~300 M  |

## Basic Usage

```python
import numpy as np
from kerasformers.models.dino import DinoViTBackbone, DinoResNetBackbone
from kerasformers.models.dino_v2 import DinoV2Backbone

# DINO V1 ViT — returns 13 intermediate feature maps (embed + 12 blocks)
model = DinoViTBackbone.from_weights("dino_vits16")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(len(features), features[-1].shape)  # 13, (1, 197, 384)

# DINO V1 ResNet — returns 5 stage feature maps
model = DinoResNetBackbone.from_weights("dino_resnet50")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(len(features), features[-1].shape)  # 5, (1, 7, 7, 2048)

# DINOv2 — returns 13 intermediate feature maps
model = DinoV2Backbone.from_weights("dinov2_vits14")
features = model(np.random.rand(1, 224, 224, 3).astype("float32"))
print(len(features), features[-1].shape)  # 13, (1, 257, 384)
```

## Loading HF fine-tunes (DINOv2)

Any HF repo whose `model_type` is `"dinov2"` (the official `facebook/dinov2-*` checkpoints or any user fine-tune) can be loaded directly via `from_weights("hf:<repo>")`. The class reads ViT dims, depth, num heads, and LayerScale value straight from the HF config; position embeddings are bicubically resampled if your `input_shape` differs from HF's training resolution. The backbone's last feature map matches HF's `last_hidden_state` (the final LayerNorm is included). For `Dinov2For*` task-head wrappers, the `dinov2.` prefix on state-dict keys is stripped automatically and any classifier head is dropped.

```python
from kerasformers.models.dino_v2 import DinoV2Backbone

# Canonical
model = DinoV2Backbone.from_weights("hf:facebook/dinov2-base")

# User fine-tune at a different resolution
model = DinoV2Backbone.from_weights(
    "hf:Jayanth2002/dinov2-base-finetuned-SkinDisease",
    input_shape=(518, 518, 3),
)
```

## Building a Classification Model on Top

```python
import keras
from kerasformers.models.dino_v2 import DinoV2Backbone

backbone = DinoV2Backbone.from_weights("dinov2_vits14")
# Take the last feature map and CLS token, then add a head
inputs = backbone.input
features = backbone.output  # list
cls_token = features[-1][:, 0]
logits = keras.layers.Dense(10, activation="softmax")(cls_token)
model = keras.Model(inputs=inputs, outputs=logits)

# Freeze the backbone, train only the head
backbone.trainable = False
```
