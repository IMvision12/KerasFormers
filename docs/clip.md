# CLIP

**Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

CLIP (Contrastive Language-Image Pre-training) is a vision + text dual-encoder trained on hundreds of millions of (image, caption) pairs with a contrastive loss. The vision side is a ViT; the text side is a small transformer with causal masking. Both encoders project to a shared embedding space, and a learnable temperature scales the cosine-similarity logits.

CLIP's similarity-matrix output makes it useful for many downstream tasks — zero-shot classification, image-text retrieval, image-image similarity, embedding extraction for diffusion models or VLMs, and (with a supervised head) standard image classification.

## Classes

Two classes are exposed, mirroring HF's `CLIP*` hierarchy:

| Class | HF equivalent | Purpose |
|---|---|---|
| `CLIPModel` | `CLIPModel` | Full contrastive dual-encoder. Image encoder + text encoder + learnable `logit_scale`. Outputs an `(B, B)` similarity matrix — use it for zero-shot classification, retrieval, or as a frozen encoder. |
| `CLIPImageClassify` | `CLIPForImageClassification` | Vision encoder only + mean-pool patches + linear classifier head. Outputs `(B, num_labels)` class logits. For supervised image classification on a fixed label set. |

Both load the same way:

```python
from kerasformers.models.clip import CLIPModel, CLIPImageClassify

# kerasformers release variant
model = CLIPModel.from_weights("clip_vit_base_16")

# Any HF Hub repo whose model_type is "clip"
model = CLIPModel.from_weights("hf:openai/clip-vit-base-patch16")
model = CLIPImageClassify.from_weights("hf:<user>/clip-finetune-imagenet")
```

## Model Variants

Variant ids for `CLIPModel.from_weights`:

| Variant id              | Params  | Patch | Resolution | Source             |
|-------------------------|--------:|------:|-----------:|--------------------|
| `clip_vit_base_16`      | ~150 M  |    16 |        224 | openai             |
| `clip_vit_base_32`      | ~151 M  |    32 |        224 | openai             |
| `clip_vit_large_14`     | ~428 M  |    14 |        224 | openai             |
| `clip_vit_large_14_336` | ~428 M  |    14 |        336 | openai             |
| `clip_vit_g_14`         | ~1.4 B  |    14 |        224 | laion2b s12B b42K  |
| `clip_vit_bigg_14`      | ~2.5 B  |    14 |        224 | laion2b 39B b160k  |

Weights are hosted on the kerasformers [`clip`](https://github.com/IMvision12/KerasFormers/releases/tag/clip) release tag and downloaded on first call.

## Loading HF Fine-tunes

Any HF repo whose `model_type` is `"clip"` (the official `openai/clip-*` checkpoints, LAION CLIP variants, or arbitrary user fine-tunes) can be loaded directly via `from_weights("hf:<repo>")`. The class reads ViT dims, depth, heads, vocab, MLP ratios, `hidden_act` (`"quick_gelu"` for OpenAI, `"gelu"`/`"gelu_new"` for LAION), and `layer_norm_eps` straight from the HF config.

```python
from kerasformers.models.clip import CLIPModel

# Canonical OpenAI
model = CLIPModel.from_weights("hf:openai/clip-vit-base-patch16")

# LAION variant — uses "gelu" instead of "quick_gelu"
model = CLIPModel.from_weights("hf:laion/CLIP-ViT-B-16-laion2B-s34B-b88K")
```

## What `CLIPModel` Returns

```python
out = model({"images": ..., "token_ids": ..., "padding_mask": ...})
out["image_logits"]   # (B, B) — image[i] vs text[j] similarity
out["text_logits"]    # (B, B) — transpose of image_logits
```

The output is a **similarity matrix**, not class probabilities. Entry `(i, j)` is:

```
image_logits[i, j] = exp(logit_scale) * cos_sim(image_embed_i, text_embed_j)
```

`B` is whatever batch you fed in: at zero-shot classification time, `B = num_class_prompts`.

## Features and Capabilities

- **Zero-Shot Classification:** Encode class names as text → compare against the image embedding → pick the nearest text. No fine-tuning required, class list can change at inference time.
- **Cross-Modal Retrieval:** Image→text and text→image search via the shared embedding space.
- **Robust Representations:** Vision and text embeddings work as drop-in features for downstream pipelines (captioning, VQA, diffusion, multimodal LLMs).
- **Supervised Classification:** Use `CLIPImageClassify` for a fixed-label classification head on top of the CLIP vision encoder.
- **HF passthrough:** `from_weights("hf:org/repo")` works for arbitrary community fine-tunes whose `model_type` is `"clip"`.

## Basic Usage — Zero-Shot Classification

```python
import keras
from kerasformers.models.clip import CLIPModel, CLIPProcessor

processor = CLIPProcessor()
model = CLIPModel.from_weights("clip_vit_base_16")

inputs = processor(text=["mountains", "tortoise", "cat"], image_paths="cat1.jpg")
output = model({
    "images": inputs["images"],
    "token_ids": inputs["input_ids"],
    "padding_mask": inputs["attention_mask"],
})

# (1, 3) — single image, 3 class prompts. Softmax over the text axis.
preds = keras.ops.softmax(output["image_logits"], axis=-1).numpy().squeeze()
result = dict(zip(["mountains", "tortoise", "cat"], preds))
print(result)
# {'mountains': 0.0006278555, 'tortoise': 0.000326458, 'cat': 0.99904567}
```

### Batch Processing Multiple Images

```python
import keras
from kerasformers.models.clip import CLIPModel, CLIPProcessor

processor = CLIPProcessor()
model = CLIPModel.from_weights("clip_vit_base_16")

image_paths = ["dog.jpg", "cat1.jpg"]
labels = ["a photo of a dog", "a photo of a car",
          "a photo of a flower", "a photo of a cat"]

inputs = processor(text=labels, image_paths=image_paths)
output = model({
    "images": inputs["images"],
    "token_ids": inputs["input_ids"],
    "padding_mask": inputs["attention_mask"],
})

probs = keras.ops.softmax(output["image_logits"], axis=-1).numpy()
for i, img_path in enumerate(image_paths):
    print(f"\nPredictions for {img_path}:")
    for j, label in enumerate(labels):
        print(f"  {label}: {probs[i, j]:.4f}")
```

## Supervised Image Classification — `CLIPImageClassify`

Mirrors HF's `CLIPForImageClassification`: the CLIP vision encoder feeds a mean-pool over the patch tokens (CLS excluded) and a single linear `classifier` Dense producing `num_labels` logits. The text tower, visual projection, and `logit_scale` are **not** built.

```python
from kerasformers.models.clip import CLIPImageClassify, CLIPImageProcessor

# Load a fine-tuned HF CLIPForImageClassification checkpoint
model = CLIPImageClassify.from_weights("hf:<user>/clip-finetuned-imagenet")
image_processor = CLIPImageProcessor()

images = image_processor("cat.jpg")
logits = model(images)               # (B, num_labels)
pred = logits.argmax(axis=-1)
```

Construct from scratch for fine-tuning on a new dataset:

```python
model = CLIPImageClassify(
    num_labels=10,                   # your class count
    input_image_shape=224,
    vision_layers=12, vision_width=768, vision_patch_size=16,
    vision_mlp_ratio=4.0,
    hidden_act="quick_gelu",         # or "gelu" / "gelu_new"
    layer_norm_eps=1e-5,
)
```

You can also warm-start the vision encoder from a `CLIPModel` checkpoint (the encoder weight names match across both classes):

```python
src = CLIPModel.from_weights("clip_vit_base_16")
ac = CLIPImageClassify(
    num_labels=10, input_image_shape=224, vision_layers=12,
    vision_width=768, vision_patch_size=16, vision_mlp_ratio=4.0,
)
# Transfer the vision encoder weights; leave the classifier random
for src_layer, dst_layer in zip(src.layers, ac.layers):
    if "vision_model" in dst_layer.name and src_layer.name == dst_layer.name:
        dst_layer.set_weights(src_layer.get_weights())
```

## Data Format

Every processor and format-sensitive post-processor in this module accepts a `data_format=None` kwarg. The default (`None`) resolves to `keras.config.image_data_format()`; pass `"channels_first"` or `"channels_last"` to override per-call without touching global state.

```python
processor = CLIPImageProcessor(data_format="channels_first")
inputs = processor("photo.jpg")
```

Image processors return tensors in the requested layout; post-processors accept tensors in either layout and read the flag to pick the channel axis. See `docs/utils.md` for which families have format-sensitive post-processors.
