# DeepSeek-VL

DeepSeek's first vision-language models, ported to pure Keras 3. A SigLIP tower
with exact-gelu (matching the reference) and an MLP connector feed a DeepSeek
text decoder. Each image expands to `num_image_tokens` placeholders.

Links:

- Paper: [DeepSeek-VL: Towards Real-World Vision-Language Understanding (arXiv:2403.05525)](https://arxiv.org/abs/2403.05525)

See also [deepseek_vl_hybrid.md](deepseek_vl_hybrid.md), [janus.md](janus.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `deepseek_vl_1.3b_chat` | kerasformers release |
| `deepseek_vl_1.3b_base` | kerasformers release |

## API

### `DeepseekVLModel`

DeepSeek-VL multimodal backbone: SigLIP tower + 2-linear GELU aligner + Llama decoder.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `102400` | token vocabulary size |
| `embed_dim` | `2048` | text model width |
| `mlp_dim` | `5632` | MLP inner width |
| `num_layers` | `24` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `16` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `norm_eps` | `1e-06` | normalization epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |
| `tie_embeddings` | `False` | reuse embeddings as the LM head |
| `vision_embed_dim` | `1024` | vision tower width |
| `vision_mlp_dim` | `4096` | vision MLP width |
| `vision_num_layers` | `24` | vision tower depth |
| `vision_num_heads` | `16` | vision attention heads |
| `image_size` | `384` | expected image resolution |
| `patch_size` | `16` | patch size |
| `vision_norm_eps` | `1e-06` | vision tower norm epsilon |
| `image_token_id` | `100015` | placeholder token id expanded per image |

### `DeepseekVLGenerate`

DeepSeek-VL with an LM head + fast ``.generate()`` (image+text -> text).

```python
generate(input_ids, attention_mask=None, max_new_tokens=None,
         eos_token_id=None, sampler=None, seed=None, **prefill_inputs)
```

Image and video tensors ride along as `**prefill_inputs`; the processor
produces them for you.

### `DeepseekVLVisionModel`

SigLIP vision tower: biased conv patch embed + learned position embeddings -> pre-LN encoder blocks (exact gelu) -> final LayerNorm.

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | required | text model width |
| `mlp_dim` | required | MLP inner width |
| `num_layers` | required | decoder blocks |
| `num_heads` | required | query heads |
| `image_size` | `384` | expected image resolution |
| `patch_size` | `16` | patch size |
| `norm_eps` | `1e-06` | normalization epsilon |

### `DeepseekVLImageProcessor`

Preprocess images for DeepSeek-VL.

| Arg | Default | Meaning |
|---|---|---|
| `size` | `384` | target resolution |
| `min_size` | `14` | smallest allowed edge |
| `background_color` | `(127, 127, 127)` | pad colour for letterboxing |
| `image_mean` | `(0.5, 0.5, 0.5)` | per-channel normalization mean |
| `image_std` | `(0.5, 0.5, 0.5)` | per-channel normalization std |
| `data_format` | `None` | `channels_last` or `channels_first` |

### `DeepseekVLProcessor`

Image + text -> model inputs for DeepSeek-VL.

| Arg | Default | Meaning |
|---|---|---|
| `variant` | `None` | variant whose tokenizer/processor files to fetch |
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `num_image_tokens` | `576` | tokens each image expands to |
| `tokenizer` | `None` | override the default tokenizer |
| `image_processor` | `None` | override the default image processor |

### `DeepseekVLTokenizer`

DeepSeek-VL BPE tokenizer (``tokenizers`` backend).

| Arg | Default | Meaning |
|---|---|---|
| `variant` | `None` | variant whose tokenizer/processor files to fetch |
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

## End-to-end example

### Single input (image + text)

```python
import os
os.environ["KERAS_BACKEND"] = "torch"   # or "jax" / "tensorflow"

from PIL import Image
from kerasformers.models.deepseek_vl import DeepseekVLGenerate, DeepseekVLProcessor

model = DeepseekVLGenerate.from_weights("deepseek_vl_1.3b_chat")
processor = DeepseekVLProcessor.from_weights("deepseek_vl_1.3b_chat")

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

`DeepseekVLProcessor` renders one conversation per call: it walks messages, so a list
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

`DeepseekVLTokenizer` encodes raw text: it has no chat template, so pass a prompt you
have rendered yourself (or go through the processor above).

```python
from kerasformers.models.deepseek_vl import DeepseekVLTokenizer

tokenizer = DeepseekVLTokenizer.from_weights("deepseek_vl_1.3b_chat")
inputs = tokenizer("Who wrote Dune?")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = DeepseekVLGenerate.from_weights(
    "deepseek_vl_1.3b_chat", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
