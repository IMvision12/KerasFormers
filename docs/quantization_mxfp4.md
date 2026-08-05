# mxfp4

OCP **MXFP4**: 4-bit floating point (e2m1) packed two values per `uint8`, with a shared
**e8m0** (power-of-two) scale per 32-value block. A floating-point 4-bit grid, ~4x
smaller than bf16. This is the format OpenAI's **GPT-OSS** ships its MoE experts in, and
kerasformers keeps them that way.

> **Not a `quantize_model` scheme.** Unlike [int8](quantization_int8.md) /
> [int4](quantization_int4.md) / [fp8](quantization_fp8.md), which you *apply* to any
> model, MXFP4 is a *load format*: GPT-OSS's experts arrive packed and stay packed,
> dequantized on the fly at run time. There is no `Mxfp4Config`, and
> `quantize_model(model, "mxfp4")` is not a thing. It is GPT-OSS specific.

## Usage

The GPT-OSS experts are MXFP4 already, so loading the Hub weights keeps them packed with
no flag:

```python
from kerasformers.models.gpt_oss import GptOssGenerate

model = GptOssGenerate.from_weights("kerasformers/gpt-oss-120b")  # experts stay MXFP4
```

The `mxfp4_experts` config field toggles it when you build a model yourself:

```python
from kerasformers.models.gpt_oss import GptOssConfig, GptOssGenerate

GptOssGenerate(GptOssConfig(mxfp4_experts=True))   # packed uint8 experts (hosted default)
GptOssGenerate(GptOssConfig(mxfp4_experts=False))  # experts expanded to float at build
```

## The expert layer

`GptOssMXFP4Experts` (`kerasformers/quantization/mxfp4_experts.py`) is the packed
counterpart of `GptOssExperts`. It stores `gate_up_proj` / `down_proj` as `uint8`
`_blocks` (nibble pairs) + `uint8` `_scales` (e8m0), keeps the biases full precision,
and dequantizes on the fly in `call` (weight-only: memory, not speed), consuming the
result in its natural `(E, 2I, H)` / `(E, H, I)` layout so no transpose is needed and the
block tensors map one-to-one onto the checkpoint.

It lives in the quantization package rather than the model file, mirroring how
transformers keeps its `Mxfp4GptOssExperts` in `integrations/mxfp4.py`.

## Primitives

`kerasformers/quantization/mxfp4_quantize.py` holds the pure-`keras.ops`,
backend-agnostic pack / unpack:

```python
from kerasformers.quantization.mxfp4_quantize import quantize_to_mxfp4, dequantize_mxfp4

blocks, scales = quantize_to_mxfp4(w)        # float -> packed (uint8 blocks + e8m0 scales)
w_approx = dequantize_mxfp4(blocks, scales)  # packed -> float
```

- **`dequantize_mxfp4(blocks, scales, dtype="float32")`**: nibble -> FP4 codebook, times
  `2^(e8m0 - 127)`. A bit-exact port of HF's `convert_moe_packed_tensors` (validated max
  \|Δ\| **0.0**), so a repo's packed experts decode to exactly the reference values.
- **`quantize_to_mxfp4(w)`**: the inverse. Picks the e8m0 block scale
  (`floor(log2(amax)) - 2`, OCP MXFP4) and rounds each value to the nearest FP4 grid
  point. A value-exact inverse on the lattice (round-trip **0.0**).

Both run on **every backend including CPU**. transformers' `quantize_to_mxfp4` needs the
GPU triton kernel; the pure-Keras version here does not.

## Footprint

4 bits per expert weight, so the experts are ~4x smaller than a bf16 copy: GPT-OSS-120B's
experts are ~57 GB packed (vs ~230 GB in bf16), keeping the whole checkpoint near the
official ~66 GB. Weight-only, so this is a memory win, not a speed one, since the dequant
runs each forward.

See [Quantization](quantization.md) for the general int8 / int4 / fp8 machinery, and
[gpt_oss.md](gpt_oss.md) for the model itself.
