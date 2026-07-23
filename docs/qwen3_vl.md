# Qwen3-VL

<div style="background:#fdecea; border:1px solid #f5c6c0; border-radius:3px; padding:12px 16px; color:#4a2626;">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored on the kerasformers
release page. <code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/" style="color:#1a5c8a;">Loading Weights</a>.
</div>
<br>

Alibaba's Qwen3-VL vision-language models, ported to pure Keras 3. It follows the
Qwen-VL line (native-resolution ViT, M-RoPE decoder) with a 16px patch and its
own video processor, which applies a clip-level frame-count-aware resize budget
rather than Qwen2-VL's per-frame budget, and samples frames at 2 fps by default.

Links:

- HF docs: [transformers/model_doc/qwen3_vl](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)

See also [qwen2_vl.md](qwen2_vl.md), [qwen2_5_vl.md](qwen2_5_vl.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `qwen3-vl-2b-instruct` | [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| `qwen3-vl-2b-thinking` | [`Qwen/Qwen3-VL-2B-Thinking`](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking) |
| `qwen3-vl-4b-instruct` | [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| `qwen3-vl-4b-thinking` | [`Qwen/Qwen3-VL-4B-Thinking`](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking) |
| `qwen3-vl-8b-instruct` | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| `qwen3-vl-8b-thinking` | [`Qwen/Qwen3-VL-8B-Thinking`](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) |
| `qwen3-vl-32b-instruct` | [`Qwen/Qwen3-VL-32B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| `qwen3-vl-32b-thinking` | [`Qwen/Qwen3-VL-32B-Thinking`](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking) |

## API

### `Qwen3VLModel`

Qwen3-VL multimodal backbone: vision tower + Qwen3 decoder + DeepStack.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `2048` | text model width |
| `mlp_dim` | `6144` | MLP inner width |
| `num_layers` | `28` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `norm_eps` | `1e-06` | normalization epsilon |
| `rope_theta` | `5000000.0` | rotary base frequency |
| `mrope_section` | `(24, 20, 20)` | M-RoPE split across time/height/width |
| `tie_embeddings` | `True` | reuse embeddings as the LM head |
| `vision_depth` | `24` | vision tower depth |
| `vision_embed_dim` | `1024` | vision tower width |
| `vision_mlp_dim` | `4096` | vision MLP width |
| `vision_num_heads` | `16` | vision attention heads |
| `vision_out_dim` | `None` | projector output width (matches the decoder) |
| `vision_act` | `'gelu_pytorch_tanh'` |  |
| `num_position_embeddings` | `2304` | learned position grid size |
| `deepstack_visual_indexes` | `(5, 11, 17)` | vision blocks that feed a DeepStack merger |
| `patch_size` | `16` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `in_channels` | `3` | input image channels |
| `image_token_id` | `151655` | placeholder token id expanded per image |
| `video_token_id` | `151656` | placeholder token id expanded per video |
| `vision_start_token_id` | `151652` | token id opening a vision span |
| `vision_end_token_id` | `151653` | token id closing a vision span |

### `Qwen3VLGenerate`

Qwen3-VL with an LM head + fast ``.generate()`` (image+text -> text).

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `2048` | text model width |
| `mlp_dim` | `6144` | MLP inner width |
| `num_layers` | `28` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `norm_eps` | `1e-06` | normalization epsilon |
| `rope_theta` | `5000000.0` | rotary base frequency |
| `mrope_section` | `(24, 20, 20)` | M-RoPE split across time/height/width |
| `tie_embeddings` | `True` | reuse embeddings as the LM head |
| `vision_depth` | `24` | vision tower depth |
| `vision_embed_dim` | `1024` | vision tower width |
| `vision_mlp_dim` | `4096` | vision MLP width |
| `vision_num_heads` | `16` | vision attention heads |
| `vision_out_dim` | `None` | projector output width (matches the decoder) |
| `vision_act` | `'gelu_pytorch_tanh'` |  |
| `num_position_embeddings` | `2304` | learned position grid size |
| `deepstack_visual_indexes` | `(5, 11, 17)` | vision blocks that feed a DeepStack merger |
| `patch_size` | `16` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `in_channels` | `3` | input image channels |
| `image_token_id` | `151655` | placeholder token id expanded per image |
| `video_token_id` | `151656` | placeholder token id expanded per video |
| `vision_start_token_id` | `151652` | token id opening a vision span |
| `vision_end_token_id` | `151653` | token id closing a vision span |

```python
generate(input_ids, attention_mask=None, max_new_tokens=None,
         eos_token_id=None, sampler=None, seed=None, **prefill_inputs)
```

Image and video tensors ride along as `**prefill_inputs`; the processor
produces them for you.

### `Qwen3VLTextModel`

Qwen3 causal decoder with DeepStack visual-feature injection.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | required | token vocabulary size |
| `embed_dim` | required | text model width |
| `mlp_dim` | required | MLP inner width |
| `num_layers` | required | decoder blocks |
| `num_heads` | required | query heads |
| `num_kv_heads` | required | key/value heads (GQA) |
| `head_dim` | required | per-head width |
| `norm_eps` | `1e-06` | normalization epsilon |

### `Qwen3VLVisionModel`

Qwen3-VL vision tower: learned pos-embeds -> GELU blocks -> merger + DeepStack.

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | required | text model width |
| `depth` | required | vision tower depth |
| `num_heads` | required | query heads |
| `intermediate_size` | required | MLP inner width |
| `out_hidden_size` | required | projector output width |
| `num_position_embeddings` | required | learned position grid size |
| `deepstack_visual_indexes` | required | vision blocks that feed a DeepStack merger |
| `hidden_act` | `'gelu_pytorch_tanh'` |  |
| `patch_size` | `16` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |

### `Qwen3VLProcessor`

Qwen3-VL image/video+text processor: like :class:`Qwen2VLProcessor` but with a 16px patch and the Qwen3-VL video processor (``[0.5]*3`` normalization and a clip-level resize budget).

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `'Qwen/Qwen3-VL-2B-Instruct'` | Hub repo to pull tokenizer/processor files from |
| `patch_size` | `16` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |

### `Qwen3VLVideoProcessor`

Qwen3-VL video processor: like :class:`Qwen2VLVideoProcessor` but with a 16px patch, ``[0.5, 0.5, 0.5]`` mean/std, and a clip-level (frame-count-aware) resize budget. The flattened patch layout is identical, so the shared vision tower consumes the output unchanged. Pixel values are assumed in ``[0, 255]``.

| Arg | Default | Meaning |
|---|---|---|
| `patch_size` | `16` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `min_pixels` | `131072` | smallest allowed pixel budget |
| `max_pixels` | `786432` | largest allowed pixel budget |
| `image_mean` | `(0.5, 0.5, 0.5)` | per-channel normalization mean |
| `image_std` | `(0.5, 0.5, 0.5)` | per-channel normalization std |
| `do_sample_frames` | `True` | subsample frames from the clip |
| `fps` | `2` | frames per second to sample |
| `num_frames` | `None` | frames to sample per clip |
| `min_frames` | `4` | fewest frames per clip |
| `max_frames` | `768` | most frames per clip |

## End-to-end example

### Single input (image + text)

```python
import os
os.environ["KERAS_BACKEND"] = "torch"   # or "jax" / "tensorflow"

from PIL import Image
from kerasformers.models.qwen3_vl import Qwen3VLGenerate, Qwen3VLProcessor

model = Qwen3VLGenerate.from_weights("qwen3-vl-2b-instruct")
processor = Qwen3VLProcessor.from_weights("qwen3-vl-2b-instruct")

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

Pass a list of conversations. Each one is rendered separately and takes only
the images its own markers claim, so the conversations do not need the same
number of images or images of the same size:

```python
conversations = [
    [{"role": "user", "content": [
        {"type": "image", "image": Image.open("a.jpg")},
        {"type": "text", "text": "What is in this image?"}]}],
    [{"role": "user", "content": [
        {"type": "image", "image": Image.open("b.jpg")},
        {"type": "image", "image": Image.open("c.jpg")},
        {"type": "text", "text": "What differs between these?"}]}],
]
inputs = processor(conversation=conversations)
outputs = model.generate(**inputs, max_new_tokens=64)

for text in processor.batch_decode(outputs):
    print(text)
```

Text-only prompts batch the same way: pass `text=[...]` with no `images`.

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen3VLGenerate.from_weights(
    "qwen3-vl-2b-instruct", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
