# Cohere 2 Vision

Cohere's vision-language model, ported to pure Keras 3. A vision tower with
dynamic image tiling (`min_patches`..`max_patches` plus an optional thumbnail)
and a pixel-shuffle downsampler (`downsample_factor`) feed the Cohere 2 text
decoder, which keeps mean-centered LayerNorm and parallel attention/MLP.

Unlike the text-only Cohere models, the VLM head does not apply `logit_scale`.


See also [cohere2.md](cohere2.md), [cohere2_moe.md](cohere2_moe.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `command-a-vision-07-2025` | [`CohereLabs/command-a-vision-07-2025`](https://huggingface.co/CohereLabs/command-a-vision-07-2025) |

## API

### `Cohere2VisionModel`

Cohere2-Vision (Command-A Vision) multimodal backbone.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `256000` | token vocabulary size |
| `embed_dim` | `4096` | text model width |
| `num_layers` | `32` | decoder blocks |
| `num_heads` | `32` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `mlp_dim` | `14336` | MLP inner width |
| `sliding_window` | `4096` | local attention span |
| `sliding_window_pattern` | `4` | one global layer every N |
| `norm_eps` | `1e-05` | normalization epsilon |
| `rope_theta` | `50000.0` | rotary base frequency |
| `attention_bias` | `False` | add bias to the QKV projections |
| `logit_scale` | `0.25` |  |
| `tie_embeddings` | `True` | reuse embeddings as the LM head |
| `vision_embed_dim` | `1152` | vision tower width |
| `vision_mlp_dim` | `4304` | vision MLP width |
| `vision_num_layers` | `27` | vision tower depth |
| `vision_num_heads` | `16` | vision attention heads |
| `image_size` | `512` | expected image resolution |
| `patch_size` | `16` | patch size |
| `vision_norm_eps` | `1e-06` | vision tower norm epsilon |
| `downsample_factor` | `2` | pixel-shuffle downsample factor |
| `alignment_intermediate_size` | `36864` |  |
| `image_token_id` | `255036` | placeholder token id expanded per image |

### `Cohere2VisionGenerate`

Cohere2-Vision (Command-A Vision) with an LM head + fast ``.generate()`` (image+text -> text).

```python
generate(input_ids, attention_mask=None, max_new_tokens=None,
         eos_token_id=None, sampler=None, seed=None, **prefill_inputs)
```

Image and video tensors ride along as `**prefill_inputs`; the processor
produces them for you.

### `Cohere2VisionImageProcessor`

InternVL dynamic-tiling image processor (HF GotOcr2 recipe).

| Arg | Default | Meaning |
|---|---|---|
| `size` | `512` | target resolution |
| `min_patches` | `1` | fewest tiles per image |
| `max_patches` | `12` | most tiles per image |
| `crop_to_patches` | `True` | tile the image dynamically by aspect ratio |
| `use_thumbnail` | `True` | append a whole-image thumbnail tile |
| `image_mean` | `(0.5, 0.5, 0.5)` | per-channel normalization mean |
| `image_std` | `(0.5, 0.5, 0.5)` | per-channel normalization std |

### `Cohere2VisionProcessor`

Image + text -> model inputs for Cohere2-Vision (Command-A Vision).

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `'CohereLabs/command-a-vision-07-2025'` | Hub repo to pull tokenizer/processor files from |
| `size` | `512` | target resolution |
| `patch_size` | `16` | patch size |
| `downsample_factor` | `2` | pixel-shuffle downsample factor |
| `image_token` | `'<image>'` |  |
| `tokenizer` | `None` | override the default tokenizer |
| `image_processor` | `None` | override the default image processor |

## End-to-end example

### Single input (image + text)

`Cohere2VisionProcessor` takes an already-rendered prompt rather than a message list:
it does not apply a chat template. The prompt must carry one
`<image>` marker per image, which the processor expands
into the right number of patch tokens.

```python
import os
os.environ["KERAS_BACKEND"] = "torch"   # or "jax" / "tensorflow"

from PIL import Image
from kerasformers.models.cohere2_vision import Cohere2VisionGenerate, Cohere2VisionProcessor

model = Cohere2VisionGenerate.from_weights("command-a-vision-07-2025")
processor = Cohere2VisionProcessor.from_weights("command-a-vision-07-2025")

image = Image.open("photo.jpg")
prompt = "<image>Describe this image in one sentence."
inputs = processor(text=prompt, images=[image])
outputs = model.generate(**inputs, max_new_tokens=64)

print(processor.decode(outputs[0]))
```

The tokenizer prepends `<BOS_TOKEN>` for you, but no turn markers: for
instruction-following output, wrap the prompt in the turn markers from the
checkpoint's chat template on the Hub.

### Batch

Pass a list of prompts and the matching images. Each prompt takes the images
its own markers claim, the processor pads them, and `generate` runs the batch
together:

```python
prompts = ["<image>What is in this image?", "<image>Describe the colours."]
images = [Image.open("a.jpg"), Image.open("b.jpg")]
inputs = processor(text=prompts, images=images)
outputs = model.generate(**inputs, max_new_tokens=64)

for text in processor.batch_decode(outputs):
    print(text)
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Cohere2VisionGenerate.from_weights(
    "command-a-vision-07-2025", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
