# LocateAnything

NVIDIA's LocateAnything-3B grounding model, ported to pure Keras 3. A MoonViT
native-resolution vision tower and connector feed a Qwen2.5-3B decoder, targeting
vision-language grounding: object detection, OCR, pointing and referring.

The learned position grid uses the same spelled-out bicubic interpolation as
Kimi's MoonViT, for backend-consistent parity.


See also [kimi_k25.md](kimi_k25.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `locateanything_3b` | kerasformers release |

## API

### `LocateAnythingModel`

LocateAnything-3B backbone (no LM head).

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `152681` | token vocabulary size |
| `embed_dim` | `2048` | text model width |
| `mlp_dim` | `11008` | MLP inner width |
| `num_layers` | `36` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `norm_eps` | `1e-06` | normalization epsilon |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `tie_embeddings` | `True` | reuse embeddings as the LM head |
| `vision_embed_dim` | `1152` | vision tower width |
| `vision_depth` | `27` | vision tower depth |
| `vision_num_heads` | `16` | vision attention heads |
| `vision_mlp_dim` | `4304` | vision MLP width |
| `vision_patch_size` | `14` | vision patch size |
| `vision_init_pos_h` | `64` |  |
| `vision_init_pos_w` | `64` |  |
| `merge_kernel` | `(2, 2)` | patch-merge kernel |
| `vision_rope_theta` | `10000.0` | rotary base in the vision tower |
| `image_token_index` | `151665` |  |
| `block_size` | `6` |  |
| `max_position_embeddings` | `32768` | longest position index the model builds |

### `LocateAnythingGenerate`

LocateAnything-3B with the (tied) Qwen2 LM head -> logits.

```python
generate(input_ids, attention_mask=None, max_new_tokens=None,
         eos_token_id=None, sampler=None, seed=None, **prefill_inputs)
```

Image and video tensors ride along as `**prefill_inputs`; the processor
produces them for you.

### `LocateAnythingVisionModel`

MoonViT-SO-400M: native-resolution packed ViT.

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | `1152` | text model width |
| `depth` | `27` | vision tower depth |
| `num_heads` | `16` | query heads |
| `mlp_dim` | `4304` | MLP inner width |
| `patch_size` | `14` | patch size |
| `init_pos_h` | `64` |  |
| `init_pos_w` | `64` |  |
| `merge_kernel` | `(2, 2)` | patch-merge kernel |
| `in_channels` | `3` | input image channels |
| `rope_theta` | `10000.0` | rotary base frequency |

### `LocateAnythingTokenizer`

Qwen2.5 BPE tokenizer extended with LocateAnything's grounding tokens.

| Arg | Default | Meaning |
|---|---|---|
| `variant` | `None` | variant whose tokenizer/processor files to fetch |
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

### `LocateAnythingImageProcessor`

Native-resolution patch preprocessor for LocateAnything / MoonViT.

| Arg | Default | Meaning |
|---|---|---|
| `patch_size` | `14` | patch size |
| `image_mean` | `(0.5, 0.5, 0.5)` | per-channel normalization mean |
| `image_std` | `(0.5, 0.5, 0.5)` | per-channel normalization std |
| `in_token_limit` | `4096` | max patches per image |
| `merge_kernel_size` | `(2, 2)` | patch-merge kernel |

### `LocateAnythingProcessor`

Image + text -> model inputs for LocateAnything-3B.

| Arg | Default | Meaning |
|---|---|---|
| `variant` | `None` | variant whose tokenizer/processor files to fetch |
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer` | `None` | override the default tokenizer |
| `image_processor` | `None` | override the default image processor |
| `merge_kernel_size` | `(2, 2)` | patch-merge kernel |

## End-to-end example

### Single input (image + text)

```python
import os
os.environ["KERAS_BACKEND"] = "torch"   # or "jax" / "tensorflow"

from PIL import Image
from kerasformers.models.locateanything import LocateAnythingGenerate, LocateAnythingProcessor

model = LocateAnythingGenerate.from_weights("locateanything_3b")
processor = LocateAnythingProcessor.from_weights("locateanything_3b")

image = Image.open("photo.jpg")
inputs = processor(conversation=[{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image in one sentence."},
    ],
}])
outputs = model.generate(**inputs, max_new_tokens=64)

print(processor.decode(outputs[0]))
```

### Several images in one conversation

Add one image content item per image. The processor expands each marker to
that image's own patch count:

```python
inputs = processor(conversation=[{
    "role": "user",
    "content": [
        {"type": "image", "image": Image.open("a.jpg")},
        {"type": "image", "image": Image.open("b.jpg")},
        {"type": "text", "text": "What differs between these two images?"},
    ],
}])
outputs = model.generate(**inputs, max_new_tokens=64)
```

### Batch

`LocateAnythingProcessor` renders one conversation per call: it walks messages, so a list
of conversations is not a valid input. Loop over them instead:

```python
questions = [
    ("a.jpg", "What is in this image?"),
    ("b.jpg", "Describe the colours."),
]
for path, question in questions:
    inputs = processor(conversation=[{
        "role": "user",
        "content": [
            {"type": "image", "image": Image.open(path)},
            {"type": "text", "text": question},
        ],
    }])
    outputs = model.generate(**inputs, max_new_tokens=64)
    print(processor.decode(outputs[0]))
```

Text-only prompts do batch in one call, since there are no images to line
up: pass `text=[...]` with no `images`.

### Text only

```python
from kerasformers.models.locateanything import LocateAnythingTokenizer

tokenizer = LocateAnythingTokenizer.from_weights("locateanything_3b")
inputs = tokenizer([{"role": "user", "content": "Who wrote Dune?"}])
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = LocateAnythingGenerate.from_weights(
    "locateanything_3b", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
