# Qwen-VL (Qwen2-VL · Qwen2.5-VL · Qwen3-VL)

**Papers**:
[Qwen2-VL](https://arxiv.org/abs/2409.12191) ·
[Qwen2.5-VL](https://arxiv.org/abs/2502.13923) ·
[Qwen3](https://arxiv.org/abs/2505.09388)

Qwen-VL is Alibaba's family of image+video+text multimodal LLMs: a ViT-style
vision tower processes images at native resolution into patch embeddings, which
are merged (2×2) and spliced into the token-embedding sequence at
`<|image_pad|>` placeholder positions, and a Qwen causal LLM decodes text while
attending across modalities. Positions use **M-RoPE** (multimodal rotary): text
tokens get standard 1-D positions, vision tokens get 3-D temporal/height/width
positions.

kerasformers ships **pure Keras 3** ports of all three generations with
bit-close parity to HuggingFace's reference implementation (verified on the
real checkpoints — see below). The vision tower, Qwen decoder, M-RoPE fusion,
and a greedy `generate` loop run unmodified on TensorFlow / Torch / JAX.

## On-the-fly weight loading

These models are **not** uploaded as kerasformers release weights; they convert
on the fly from the Hugging Face checkpoints (safetensors are downloaded and
mapped to Keras at load time):

```python
from kerasformers.models.qwen2_vl import Qwen2VLModel
from kerasformers.models.qwen2_5_vl import Qwen2_5_VLModel
from kerasformers.models.qwen3_vl import Qwen3VLModel

m = Qwen2VLModel.from_weights("hf:Qwen/Qwen2-VL-2B-Instruct")
m = Qwen2_5_VLModel.from_weights("hf:Qwen/Qwen2.5-VL-3B-Instruct")
m = Qwen3VLModel.from_weights("hf:Qwen/Qwen3-VL-2B-Instruct")
```

Any HF repo whose `config.json` `model_type` matches (`qwen2_vl` / `qwen2_5_vl`
/ `qwen3_vl`) loads this way, including community fine-tunes. bf16 checkpoints
are cast to float32 on transfer; tied and untied LM heads are both handled.

## Verified parity

Each port was validated against the HF reference (forced eager attention) on a
real image+text forward pass — **argmax agreement 1.0000** at every position:

| Model | Checkpoint | max \|Δ logits\| | argmax match |
|---|---|---|---|
| Qwen2-VL | `Qwen/Qwen2-VL-2B-Instruct` | 7.3e-4 | 1.0000 |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct` | 4.1e-4 | 1.0000 |
| Qwen3-VL | `Qwen/Qwen3-VL-2B-Instruct` | 3.9e-3 | 1.0000 |

Every primitive (RMSNorm, GQA + M-RoPE attention, SwiGLU, vision rotary, KV
cache) also matches HF to ~1e-7 in isolation.

## Forward pass

The model takes a dict and returns logits (plus the final hidden state):

```python
out = model({
    "input_ids":      input_ids,       # (B, L) int, image placeholders inside
    "pixel_values":   pixel_values,    # (num_patches, patch_dim) flattened patches
    "image_grid_thw": image_grid_thw,  # (num_images, 3) per-image (t, h, w)
})
out["logits"]              # (B, L, vocab_size)
```

> **Layout note** — the processor pre-flattens each patch to a
> `in_channels · temporal_patch_size · patch_size²` vector, so the model has no
> spatial H/W axes and is **layout-agnostic** (channels_first/last does not
> apply — handled like the audio models). The patch-embed Conv3d (kernel ==
> stride) is implemented as a `Dense`.

## Generation

`.generate()` does greedy multimodal decoding with a KV cache and incremental
M-RoPE positions (each new token's position is `cache_len + rope_delta` on all
three axes):

```python
ids = model.generate(input_ids, pixel_values=pv, image_grid_thw=grid,
                     max_new_tokens=128)
```

Qwen3-VL injects its **DeepStack** vision features into the first few decoder
layers during the prefill (decode steps then read them from the KV cache).

## Image processor

`Qwen2VLImageProcessor` is a pure-Python port of HF's: smart-resize each image
so both sides are multiples of `patch_size · spatial_merge_size` (pixels kept in
range), CLIP-normalize, repeat the frame to fill `temporal_patch_size`, and
reshape into the `(num_patches, patch_dim)` layout with a matching
`image_grid_thw`. Grids match HF exactly; pixels match to a small bicubic
tolerance.

```python
from kerasformers.models.qwen2_vl.qwen2_vl_image_processor import Qwen2VLImageProcessor
feat = Qwen2VLImageProcessor()(pil_image)   # {"pixel_values", "image_grid_thw"}
```

## What differs between the generations

| | Qwen2-VL | Qwen2.5-VL | Qwen3-VL |
|---|---|---|---|
| Vision norm / MLP | LayerNorm / GELU | RMSNorm / SwiGLU | LayerNorm / GELU |
| Vision attention | full | **windowed** (+ full at some layers) | full |
| Vision positions | 2-D rotary | 2-D rotary | 2-D rotary + **learned** (interpolated) |
| Extra vision | — | `tokens_per_second` (video) | **DeepStack** fusion |
| Patch size | 14 | 14 | 16 |
| Text decoder | Qwen2 (qkv bias) | Qwen2.5 (qkv bias) | **Qwen3** (QK-norm, no qkv bias) |
| M-RoPE | sectioned | sectioned | **interleaved** |

The shared LLM primitives live in `qwen2_vl/qwen2_vl_layers.py`; Qwen2.5-VL and
Qwen3-VL import and subclass from Qwen2-VL (mirroring how SigLIP2 builds on
SigLIP).

## Citation

```bibtex
@article{Qwen2VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and others}, journal={arXiv:2409.12191}, year={2024}
}
@article{Qwen2.5-VL,
  title={Qwen2.5-VL Technical Report},
  author={Bai, Shuai and others}, journal={arXiv:2502.13923}, year={2025}
}
```
