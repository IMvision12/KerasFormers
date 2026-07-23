# Qwen (text & vision-language)

<div style="background:#fdecea; border:1px solid #f5c6c0; border-radius:3px; padding:12px 16px; color:#4a2626;">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored on the kerasformers
release page. <code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/" style="color:#1a5c8a;">Loading Weights</a>.
</div>
<br>

Alibaba's Qwen family in **pure Keras 3**: both the text LLMs and the
image+video+text multimodal LLMs: with bit-close parity to HuggingFace
(verified on real checkpoints, see below). One implementation per family runs
unmodified on **TensorFlow / Torch / JAX**.

**Papers**:
[Qwen2](https://arxiv.org/abs/2407.10671) ·
[Qwen3](https://arxiv.org/abs/2505.09388) ·
[Qwen2-VL](https://arxiv.org/abs/2409.12191) ·
[Qwen2.5-VL](https://arxiv.org/abs/2502.13923)

| Family | Module | Kind | Text decoder |
|---|---|---|---|
| Qwen2 | `kerasformers.models.qwen2` | text | Qwen2 (GQA, **qkv bias**, 1-D RoPE) |
| Qwen3 | `kerasformers.models.qwen3` | text | Qwen3 (**QK-norm**, no qkv bias) |
| Qwen3.5 | `kerasformers.models.qwen3_5` | text | **Qwen3-Next hybrid** (Gated-DeltaNet + gated full attention) |
| Qwen2-VL | `kerasformers.models.qwen2_vl` | image+video+text | Qwen2 |
| Qwen2.5-VL | `kerasformers.models.qwen2_5_vl` | image+video+text | Qwen2.5 (windowed vision) |
| Qwen3-VL | `kerasformers.models.qwen3_vl` | image+video+text | Qwen3 (interleaved M-RoPE, DeepStack) |

## Loading

Each family exposes two classes:

- **`*Model`**: base model; its `call` returns features (`last_hidden_state`).
- **`*Generate`**: adds the LM head + greedy `.generate()`; `call` returns `logits`.

Weights convert **on the fly** from the public Hugging Face checkpoints
(safetensors downloaded and mapped at load time: bf16 cast to float32, tied and
untied LM heads both handled). The canonical path is the **friendly variant
name**; a raw `hf:` id also works for any matching `model_type`:

```python
from kerasformers.models.qwen3 import Qwen3Generate
from kerasformers.models.qwen2_vl import Qwen2VLGenerate

gen = Qwen3Generate.from_weights("qwen3-4b")                 # text
gen = Qwen2VLGenerate.from_weights("qwen2-vl-7b-instruct")   # multimodal
# raw hf: ids still work:
gen = Qwen3Generate.from_weights("hf:Qwen/Qwen3-4B")
```

### Available variants

Text:

| Family | Variants (`from_weights("…")`) |
|---|---|
| Qwen2 | `qwen2-{0.5b,1.5b,7b,72b}` and each `-instruct` |
| Qwen3 | `qwen3-{0.6b,1.7b,4b,8b,14b}` and each `-base`; `qwen3-32b` |
| Qwen3.5 | `qwen3.5-{0.8b,2b,4b,9b}` and each `-base`; `qwen3.5-27b` |

Vision-language:

| Family | Variants |
|---|---|
| Qwen2-VL | `qwen2-vl-{2b,7b,72b}` and each `-instruct` |
| Qwen2.5-VL | `qwen2.5-vl-{3b,7b,32b,72b}-instruct` (instruct-only series) |
| Qwen3-VL | `qwen3-vl-{2b,4b,8b,32b}-instruct` and each `-thinking` |

> **Not yet supported: Mixture-of-Experts.** The MoE checkpoints
> (Qwen2-57B-A14B `qwen2_moe`; Qwen3 30B-A3B / 235B-A22B `qwen3_moe`; Qwen3-VL
> 30B-A3B / 235B-A22B `qwen3_vl_moe`; Qwen3.5 35B-A3B / 122B-A10B / 397B-A17B
> `qwen3_5_moe_text`) use sparse expert blocks the dense ports can't load. They
> need a separate MoE implementation. Quantized repos (`-AWQ`, `-GPTQ-*`, GGUF)
> are also out of scope (the converter reads bf16/fp safetensors).

> **Qwen3.5 is itself a multimodal series.** The released checkpoints
> (`Qwen3_5ForConditionalGeneration`) bundle a vision tower; this port is the
> **text backbone** (`model_type` `qwen3_5` / `qwen3_5_text`), loaded from each
> checkpoint's `model.language_model.*` tensors (vision + MTP head ignored).

## Verified parity

Validated against the HF reference (eager attention) on a real forward pass:
**argmax agreement 1.0000** at every position; text generation is **token-exact**
greedy:

| Model | Checkpoint | max \|Δ logits\| | argmax |
|---|---|---|---|
| Qwen2 | `Qwen/Qwen2-0.5B` | 3.1e-5 | 1.0000 |
| Qwen3 | `Qwen/Qwen3-0.6B` | 2.2e-5 | 1.0000 |
| Qwen3.5 | `Qwen/Qwen3.5-0.8B` | 1.5e-5 | 1.0000 |
| Qwen2-VL | `Qwen/Qwen2-VL-2B-Instruct` | 7.3e-4 | 1.0000 |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct` | 4.1e-4 | 1.0000 |
| Qwen3-VL | `Qwen/Qwen3-VL-2B-Instruct` | 3.9e-3 | 1.0000 |

Individual primitives (RMSNorm, GQA + M-RoPE attention, SwiGLU, vision rotary,
KV cache) match HF to ~1e-7 in isolation. The Qwen3.5 residual is
chunked-vs-recurrent Gated-DeltaNet fp accumulation: the kernels are
algebraically identical.

## Forward pass

`*Model` returns features; `*Generate` adds logits. Text takes just token ids;
VL adds pre-patchified pixels:

```python
# text
gen({"input_ids": input_ids})["logits"]            # (B, L, vocab_size)

# vision-language: images and/or video; placeholders sit inside input_ids
inputs = {
    "input_ids":           input_ids,            # (B, L) int, image/video placeholders
    "pixel_values":        pixel_values,         # (num_patches, patch_dim) image patches
    "image_grid_thw":      image_grid_thw,       # (num_images, 3) per-image (t, h, w)
    "pixel_values_videos": pixel_values_videos,  # (num_patches, patch_dim) video patches
    "video_grid_thw":      video_grid_thw,       # (num_videos, 3) per-video (t, h, w)
}
gen(inputs)["logits"]                              # (B, L, vocab_size)
```

The image and video blocks are each optional. Video patches use the **same
flattened layout** as images and run through the **same vision tower**; the only
difference is `grid_t = num_frames // temporal_patch_size` (vs `1` for an image),
and their embeddings scatter into the `<|video_pad|>` slots.

These are token-id (and pre-flattened-patch) models: **no spatial H/W axes**, so
`channels_first/last` does not apply (handled like the audio models). The VL
patch-embed Conv3d (kernel == stride) is implemented as a `Dense`.

## Generation

`.generate()` is greedy decoding with a KV cache. Qwen3.5 additionally carries
the per-layer conv state + delta-rule recurrent state for its linear layers; the
VL families carry incremental M-RoPE positions (each new token's position is
`cache_len + rope_delta` on all three axes), and Qwen3-VL injects its
**DeepStack** vision features into the first decoder layers during prefill.

The API is the same shape for both: build inputs from a chat list, then
`model.generate(**inputs, max_new_tokens=...)`. **LLMs use the tokenizer**
(text only); **VLMs use the processor** (tokenizer + image / video processor),
with images or video inline in the conversation: images via `path` / `url` / a
PIL image, video via a `path` / `url` (decoded and frame-sampled automatically,
like HF) or inline `frames`.

Load the tokenizer / processor with `.from_weights(...)`, passing the **same**
identifier you give the model, so its files match the checkpoint, e.g.
`Qwen2Tokenizer.from_weights("hf:Qwen/Qwen2-7B-Instruct")` or
`Qwen2VLProcessor.from_weights("hf:Qwen/Qwen2-VL-7B-Instruct")`. A release variant
like `"qwen2-7b-instruct"` uses the family's shared tokenizer, and the bare
`Qwen2Tokenizer()` / `Qwen2VLProcessor()` constructors fall back to a default Qwen
repo.

```python
# text LLM: tokenizer takes the chat messages
from kerasformers.models.qwen3 import Qwen3Generate, Qwen3Tokenizer
model = Qwen3Generate.from_weights("qwen3-0.6b")
tokenizer = Qwen3Tokenizer.from_weights("qwen3-0.6b")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Name three prime numbers."},
]
inputs = tokenizer(messages)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))

# vision-language: processor takes the conversation (images inline)
from kerasformers.models.qwen2_vl import Qwen2VLGenerate, Qwen2VLProcessor
model = Qwen2VLGenerate.from_weights("qwen2-vl-2b-instruct")
processor = Qwen2VLProcessor.from_weights("qwen2-vl-2b-instruct")

conversation = [
    {"role": "user", "content": [
        {"type": "image", "path": "/path/to/image.jpg"},
        {"type": "text", "text": "What happened in the image?"},
    ]},
]
inputs = processor(conversation)
outputs = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## Image processor (VL)

`Qwen2VLImageProcessor` is a pure-Python port of HF's: smart-resize each image so
both sides are multiples of `patch_size · spatial_merge_size`, CLIP-normalize,
repeat the frame to fill `temporal_patch_size`, and reshape into the
`(num_patches, patch_dim)` layout with a matching `image_grid_thw`. Grids match
HF exactly; pixels match to a small bicubic tolerance.

```python
from kerasformers.models.qwen2_vl.qwen2_vl_image_processor import Qwen2VLImageProcessor
feat = Qwen2VLImageProcessor()(pil_image)   # {"pixel_values", "image_grid_thw"}
```

## Video processor (VL)

`Qwen2VLVideoProcessor` / `Qwen3VLVideoProcessor` are the pure-`keras.ops` ports of
HF's video processors. Each frame is smart-resized to a multiple of
`patch_size · spatial_merge_size`, rescaled + normalized, the frame count is padded
up to a multiple of `temporal_patch_size` (repeating the last frame), and the clip
is flattened into the **same `(num_patches, patch_dim)` layout as images** (so the
shared vision tower consumes both unchanged) with a `video_grid_thw` whose
`grid_t = num_frames // temporal_patch_size`. Grids match HF exactly; pixels match
to a small bicubic tolerance.

Qwen3-VL differs from Qwen2-VL: a 16px patch, `[0.5, 0.5, 0.5]` normalization, and a
**clip-level** resize budget (the frame count factors into the target resolution, so
more frames → smaller frames) instead of Qwen2-VL's per-frame resize.

```python
import numpy as np
from kerasformers.models.qwen2_vl import Qwen2VLVideoProcessor

frames = np.random.randint(0, 256, (8, 224, 224, 3), dtype="uint8")  # (T, H, W, C)
feat = Qwen2VLVideoProcessor()(frames)   # {"pixel_values_videos", "video_grid_thw"}
```

The processor wires this in automatically: like HF, you just point at the file. A
`{"type": "video", "path": "…"}` (or `"url"`) item is decoded by the shared
`kerasformers.utils.video.load_video` (PyAV backend; OpenCV / decord also available)
and **frame-sampled to the target fps** (Qwen3-VL: 2 fps, via the same
`num_frames = total / video_fps · fps` + `linspace` rule as HF; Qwen2-VL keeps every
frame); inline `frames` are also accepted. It produces `pixel_values_videos` /
`video_grid_thw` and expands `<|video_pad|>` to the right token count.

```python
conversation = [
    {"role": "user", "content": [
        {"type": "video", "path": "/path/to/video.mp4"},  # or "url", or inline "video": frames
        {"type": "text", "text": "What happens in the video?"},
    ]},
]
inputs = processor(conversation)
outputs = model.generate(**inputs, max_new_tokens=128)
```

> **Not yet ported:** Qwen3-VL's inter-frame timestamp tokens: the processor emits
> a single `<|vision_start|><|video_pad|><|vision_end|>` block per video (frame
> decoding + fps sampling already match HF).

## Architecture notes

### Text families

| | Qwen2 | Qwen3 | Qwen3.5 |
|---|---|---|---|
| Token mixer | GQA attention | GQA attention | **hybrid** linear / full |
| Linear attention |: |, | **Gated-DeltaNet** (conv1d + delta rule) |
| QK-norm | no | **yes** | yes (full layers) |
| QKV bias | **yes** | no | no |
| RoPE | 1-D full | 1-D full | **partial** (factor 0.25) |
| Norm | RMSNorm | RMSNorm | **zero-centered** `(1+w)` + gated |
| Output gate |, |, | **sigmoid gate** (full attention) |

For pure text, Qwen3.5's three M-RoPE position axes coincide, so rotary reduces
to standard 1-D partial rope.

### Vision-language families

| | Qwen2-VL | Qwen2.5-VL | Qwen3-VL |
|---|---|---|---|
| Vision norm / MLP | LayerNorm / GELU | RMSNorm / SwiGLU | LayerNorm / GELU |
| Vision attention | full | **windowed** (+ full at some layers) | full |
| Vision positions | 2-D rotary | 2-D rotary | 2-D rotary + **learned** (interpolated) |
| Extra vision |: | `tokens_per_second` (video) | **DeepStack** fusion |
| Patch size | 14 | 14 | 16 |
| Text decoder | Qwen2 (qkv bias) | Qwen2.5 (qkv bias) | **Qwen3** (QK-norm, no qkv bias) |
| M-RoPE | sectioned | sectioned | **interleaved** |

The shared VL primitives live in `qwen2_vl/qwen2_vl_layers.py`; Qwen2.5-VL and
Qwen3-VL subclass from Qwen2-VL. The text families are each self-contained (no
cross-family imports); every family's `convert_*_hf_to_keras.py` maps the HF
safetensors to Keras.

## Citation

```bibtex
@article{Qwen2,        title={Qwen2 Technical Report},   author={Yang, An and others}, journal={arXiv:2407.10671}, year={2024}}
@article{Qwen3,        title={Qwen3 Technical Report},   author={Yang, An and others}, journal={arXiv:2505.09388}, year={2025}}
@article{Qwen2VL,      title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution}, author={Wang, Peng and others}, journal={arXiv:2409.12191}, year={2024}}
@article{Qwen2.5-VL,   title={Qwen2.5-VL Technical Report}, author={Bai, Shuai and others}, journal={arXiv:2502.13923}, year={2025}}
```
