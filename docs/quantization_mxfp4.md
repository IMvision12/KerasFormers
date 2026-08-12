# mxfp4

OCP **MXFP4**: 4-bit floating point (e2m1) packed two values per `uint8`, with a shared
**e8m0** (power-of-two) scale per 32-value block. A floating-point 4-bit grid, ~8x smaller
than fp32 (~4x smaller than bf16). It is both the format OpenAI's **GPT-OSS** ships its
experts in, and a general `quantize_model` scheme you can apply to any model.

## As a `quantize_model` scheme

Like [int8](quantization_int8.md) / [int4](quantization_int4.md) /
[fp8](quantization_fp8.md), `"mxfp4"` is a weight-only scheme: pass it to `from_weights`
or `quantize_model` and Dense / EinsumDense / fused-expert kernels are stored packed and
dequantized on the fly. Runs on **all three backends** (pure `keras.ops`).

```python
from kerasformers.models.qwen3 import Qwen3Generate
from kerasformers.quantization import quantize_model, Mxfp4Config

# load + quantize in one call
model = Qwen3Generate.from_weights("qwen3-4b", quantization="mxfp4")  # ~8x smaller

# or quantize a built model in place
quantize_model(model, "mxfp4")
quantize_model(model, Mxfp4Config(skip_modules=("lm_head",)))  # per-method config
```

**Constraint:** MXFP4 packs *fixed* 32-value blocks, so each quantized kernel's
**contracting dimension must be a multiple of 32** (true of essentially every
transformer width / head-dim). A kernel that violates it raises a clear error; exclude it
with a `skip_modules` pattern or use `int4` (which allows any even dim). Embeddings stay
int8, like int4 (the 4-bit savings live in the Dense weights).

Accuracy is comparable to int4 (~0.98 cosine): both are ~4-bit. MXFP4 uses a *float*
(e2m1) grid with a power-of-two block scale, so it shines on weights that were **trained**
in it (GPT-OSS dequantizes bit-exact) and is a solid round-to-nearest option elsewhere.
This is not calibrated PTQ (no GPTQ / AWQ).

## As GPT-OSS's native format

GPT-OSS ships its experts in MXFP4 already, so loading its Hub weights keeps them packed
with no flag; `GptOssMXFP4Experts` holds them and dequantizes in `call`:

```python
from kerasformers.models.gpt_oss import GptOssGenerate

model = GptOssGenerate.from_weights("kerasformers/gpt-oss-120b")  # experts stay MXFP4
```

The `mxfp4_experts` config field toggles it when you build a model yourself:

```python
from kerasformers.models.gpt_oss import GptOssConfig, GptOssGenerate

GptOssGenerate(
    GptOssConfig(mxfp4_experts=True)
)  # packed uint8 experts (hosted default)
GptOssGenerate(GptOssConfig(mxfp4_experts=False))  # experts expanded to float at build
```

`GptOssMXFP4Experts` (`kerasformers/quantization/quantized_layers.py`) stores `gate_up_proj` /
`down_proj` as `uint8` `_blocks` (nibble pairs) + `uint8` `_scales` (e8m0), keeps the
biases full precision, and consumes the dequantized result in its natural `(E, 2I, H)` /
`(E, H, I)` layout so the block tensors map one-to-one onto the checkpoint. On **decode**
(single token) it dequantizes only the **top-k routed experts**, not all of them — ~8×
less dequant per layer than the dense path, at identical output; long prefills fall back to
the dense all-experts path when that is cheaper.

It lives in the quantization package rather than the model file, mirroring how transformers
keeps its `Mxfp4GptOssExperts` in `integrations/mxfp4.py`.

## Primitives

`kerasformers/quantization/mxfp4_quantize.py` holds the pure-`keras.ops`,
backend-agnostic pack / unpack and the `BaseQuantizer` that wraps them:

```python
from kerasformers.quantization import (
    MXFP4Quantizer,
    quantize_to_mxfp4,
    dequantize_mxfp4,
)

blocks, scales = quantize_to_mxfp4(w)  # float -> packed (uint8 blocks + e8m0 scales)
w_approx = dequantize_mxfp4(blocks, scales)  # packed -> float

q = MXFP4Quantizer()  # the quantize_model building block
packed, scale = q.quantize(w, axis=0)  # single packed axis (moved to the end)
w_approx = q.dequantize(packed, scale, axis=0, dtype="float32")
```

- **`dequantize_mxfp4(blocks, scales, dtype="float32")`**: nibble -> FP4 codebook, times
  `2^(e8m0 - 127)`. A bit-exact port of HF's `convert_moe_packed_tensors` (validated max
  \|Δ\| **0.0**), so a repo's packed experts decode to exactly the reference values.
- **`quantize_to_mxfp4(w)`**: the inverse. Picks the e8m0 block scale
  (`floor(log2(amax)) - 2`, OCP MXFP4) and rounds each value to the nearest FP4 grid
  point. A value-exact inverse on the lattice (round-trip **0.0**).
- **`MXFP4Quantizer`**: the `Quantizer` used by `quantize_model` — `quantize` /
  `dequantize` / `storage_spec` along an arbitrary contracting axis, `uint8` kernel +
  `uint8` (e8m0) scale.

Both run on **every backend including CPU**. transformers' `quantize_to_mxfp4` needs the
GPU triton kernel; the pure-Keras version here does not.

## Footprint

4 bits per weight, so ~4x smaller than a bf16 copy. GPT-OSS-120B's experts are ~57 GB
packed (vs ~230 GB in bf16), keeping the whole checkpoint near the official ~66 GB.
Weight-only, so this is a memory win, not a speed one, since the dequant runs each forward
(with the top-k sparse shortcut on decode).

See [Quantization](quantization.md) for the general int8 / int4 / fp8 / mxfp4 machinery,
and [gpt_oss.md](gpt_oss.md) for the model itself.
