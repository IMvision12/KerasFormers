# Quantization (int8 / int4 / fp8)

kerasformers ships its **own** weight-only int8 / int4 / fp8 quantization in
`kerasformers/quantization/`: a from-scratch, backend-agnostic implementation
(pure `keras.ops`), not Keras's built-in `model.quantize`. It shrinks a model
~4× (int8 / fp8) or ~8× (int4) so larger checkpoints fit in memory. int8 / int4
run on TensorFlow / Torch / JAX; fp8 (float8-e4m3) is torch / jax only.

## Quick start

```python
from kerasformers.models.qwen3 import Qwen3Generate

# load + quantize in one call
model = Qwen3Generate.from_weights("qwen3-4b", quantization="int8")   # ~4x smaller
model = Qwen3Generate.from_weights("qwen3-4b", quantization="int4")   # ~8x smaller
model = Qwen3Generate.from_weights("qwen3-4b", quantization="fp8")    # ~4x (torch/jax)

# or quantize a model you already built/loaded
from kerasformers.quantization import quantize_model
quantize_model(model, "int8")                 # in place
quantize_model(model, "int4", group_size=64)  # int4 block size (default 32)
quantize_model(model, "fp8")                  # float8 e4m3 (torch / jax)
```

`quantization=` is wired through `from_weights` for every model (release variants
**and** `hf:` repos).

**No-float load (for models bigger than your float budget).** Add
`low_memory=True` to stream the checkpoint straight into int storage without ever
building the full float model:

```python
# never materializes the bf16 model: quantizes each tensor as it loads
model = Qwen3Generate.from_weights("qwen3-4b", quantization="int4", low_memory=True)
```

It applies to subclassed LLMs whose converter assigns through `model.weights` (the
standard pattern) and **falls back automatically** to load-then-quantize for
anything else, so it is always safe to pass.

## Production usage

Pass a `QuantizationConfig` for fine control: named schemes, mixed precision,
and skipping accuracy-sensitive layers:

```python
from kerasformers.quantization import quantize_model, QuantizationConfig

cfg = QuantizationConfig(
    mode="int4", group_size=128,
    skip_modules=("lm_head",),               # keep these layers in float
    quantize_embeddings=True,
    overrides={"decoder_layer_0": "int8"},   # per-layer precision
)
quantize_model(model, cfg)
quantize_model(model, "int4-g128")           # or a named scheme
```

**Save / load / revert:**

```python
from kerasformers.quantization import save_quantized, load_quantized, dequantize_model

save_quantized(model, "model.weights.h5")    # int weights + ".quant.json" sidecar

skeleton = Qwen3Generate.from_weights("qwen3-4b", load_weights=False)
skeleton(dummy_inputs)                        # build the float architecture
load_quantized(skeleton, "model.weights.h5")  # replay config + load int weights

dequantize_model(model)                       # revert to float layers
```

**MoE and functional models:** `quantize_model` also quantizes **fused MoE
experts** (the `gate_up_proj` / `down_proj` banks of Qwen/GLM/DeepSeek-MoE, along
the contracted axis) and **functional / vision** models (ViT, CLIP, …). A
functional graph can't be mutated in place, so it is **cloned**: use the
returned model:

```python
qmodel = quantize_model(vit_model, "int8")    # functional -> returns a NEW model
```

## How it works

Weight-only quantization: the weights are stored quantized and **dequantized on
the fly** inside each layer's `call`, so the matmul still runs in the activation
dtype. No special int kernels are needed, which is why it is fully
backend-agnostic.

- **int8**: per-channel **symmetric absmax** (one float scale per output
  channel, over the contracting axis). `w_int8 = round(w / scale)`,
  `scale = max|w| / 127`. This is *vector-wise per-channel int8*, **not**
  LLM.int8(): there is no activation-outlier fp16 path; on very large models the
  accuracy lever is keeping outlier-heavy layers in float (`skip_modules`) or
  going group-wise int4.
- **int4**: **block-wise** symmetric absmax (the `in` axis is split into blocks
  of `group_size`, each block × output-channel gets its own scale: the
  bitsandbytes idea), packed **two values per byte**. `scale = max|w| / 7`.
- **fp8**: per-output-channel absmax cast into the native `float8_e4m3fn` dtype
  (1 byte, `scale = max|w| / 448`); the floating-point grid (4 exp / 3 mantissa
  bits) often tracks wide dynamic range better than uniform int8 at the same
  size. **torch / jax only**.
- **Embeddings**: int8 with a per-row scale; the lookup gathers int8 rows and
  dequantizes only the gathered slice (for both `int8` and `int4` model modes,
  embeddings stay int8: the 4-bit savings live in the Dense weights).

**N-D kernels.** Quantization is along the **contracting axis**, not a hardcoded
`axis=0`, so the same quantizers serve 2-D `Dense` kernels, N-D `EinsumDense`
kernels (axis derived from the equation: a tuple for int8/fp8, a single axis for
packed int4), per-row embeddings (`axis=1`), and fused MoE expert banks
(`axis=-1`). Scales keep the reduced axes as size 1 so they broadcast over any
rank with no reshape.

**Robustness.** Scales use an epsilon floor (`max(amax / MAX, ε)`) rather than an
exact-zero test (handles zero and denormal channels), and `dequantize` takes the
compute `dtype`, so `mixed_bfloat16` graphs don't upcast through float32.

`quantize_model` walks the layer tree and **swaps** every built `Dense` →
[`QuantizedDense`](../kerasformers/quantization/quantized_layers.py), `EinsumDense`
→ `QuantizedEinsumDense`, `Embedding` → `QuantizedEmbedding`, and fused experts →
`QuantizedExperts`, freeing the float weights, then records the resolved
`QuantizationConfig` on the model. The swap unlocks the keras layer tracker,
untracks the float layer, and registers the quantized one, enumerating both
`__dict__` and (on the torch backend, where keras `Layer` is an `nn.Module`)
`_modules`, so it finds sub-layers on every backend.

## Components

The package mirrors keras's `Quantizer` / `AbsMaxQuantizer` structure: a base
class plus one file per scheme:

| Symbol | File | Role |
|---|---|---|
| `Quantizer` | `base/base_quantization.py` | base class (also `kerasformers.base.Quantizer`): `quantize(weight, axis)` / `dequantize(packed, scale, axis, dtype)` / `storage_spec(weight_shape, axis)` + `get_config` / `from_config`; ships `normalize_axes` / `single_axis` |
| `Int8Quantizer` | `int8_quantize.py` | per-channel int8 quantizer (quantize / dequantize methods) |
| `Int4Quantizer` | `int4_quantize.py` | block-wise packed int4 quantizer (any axis via moveaxis; module `effective_group_size`) |
| `Fp8Quantizer` | `fp8_quantize.py` | per-channel float8-e4m3 quantizer (module `fp8_supported`; torch / jax) |
| `QuantizedDense` / `QuantizedEinsumDense` / `QuantizedEmbedding` / `QuantizedExperts` | `quantized_layers.py` | weight-only drop-in layers (each holds a quantizer); `QuantizedExperts` = fused MoE expert bank, contracting-axis quantized |
| `QuantizationConfig` / `Int8Config` / `Int4Config` / `Fp8Config` / `SCHEMES` | `quant_config.py` | recipe (mode, group_size, skip_modules, quantize_embeddings, overrides) + per-method configs + named presets |
| `quantize_model` / `quantize_functional` | `quantize.py` | in-place (subclassed) / clone (functional) model surgery |
| `quantize_skeleton` / `quantize_and_load` | `quantize.py` | no-float int skeleton / stream a float checkpoint into int storage |
| `save_quantized` / `load_quantized` / `dequantize_model` | `quantize.py` | persist (+ `.quant.json`) / reload / revert |

A `QuantizedDense` holds an `Int8Quantizer` / `Int4Quantizer` / `Fp8Quantizer`
(via `get_quantizer(mode, group_size)`) and uses it for `storage_spec` (build),
`quantize` (from a float `Dense`), and `dequantize` (in `call`).

## Accuracy & size (validated, 3 backends)

On real kerasformers decoders, output cosine vs the float model:

| Mode | Size | cosine |
|---|---|---|
| int8 | ~3.8× smaller | ~0.9999 |
| int4 (group 32) | ~5.8–8× smaller | ~0.98 |
| fp8 (e4m3) | ~3.8× smaller | ~0.9994 |

int4's ratio depends on `group_size` (bigger blocks → fewer scales → smaller, but
slightly less accurate).

## Will it fit? (memory sizing)

Weight-only quantization is about **fitting** a model, so the practical question
is bytes-per-parameter:

| precision | bytes / param | ~max params in 80 GB (weights only) |
|---|---|---|
| bf16 (float) | 2.0 | ~40B |
| int8 | ~1.0 | ~80B |
| int4 (g128) | ~0.55 | ~145B |

int4 adds the per-block fp32 scales (a few percent; a smaller `group_size` means
more scales, slightly larger). Leave **~20 % headroom** for the KV cache and
activations, so the *practical* ceilings on one 80 GB H100 are roughly **32B
bf16 / 64B int8 / ~115B int4**.

**MoE counts total, not active.** Sparse experts cut *compute* per token, but
every expert must be resident: size by total parameters, not active ones.

Worked examples (int4, ≈ 0.55 B/param):

| model | int4 weights | single 80 GB H100? |
|---|---|---|
| 70B dense | ~38 GB | yes |
| 120B (GPT-OSS-120B class) | ~66 GB | yes (tight) |
| 355B (GLM-4.5) | ~195 GB | no: ~3 GPUs |
| 744B (GLM-5.x) | ~410 GB | no: ~5–6 GPUs |

> **Load time.** By default `quantization=` builds the float model first (peak ≈
> the **bf16** size, params × 2) and quantizes after. Pass **`low_memory=True`**
> (or call `quantize_and_load`) to take the **no-float** path: an int skeleton is
> built and each tensor is quantized as it loads, so peak ≈ the *quantized* size +
> one layer's float. That is what lets a checkpoint larger than your float budget
> load quantized. It covers subclassed LLMs with the standard
> `model.weights`-iteration converter; other models fall back to the float path.

## Caveats (honest)

- **Portable weight-only = memory, not speed.** The default Keras path
  dequantizes weights to float every `call`, so it reduces footprint rather than
  latency.
- **Float path vs no-float path.** By default `quantization=` and `load_quantized`
  build the float architecture before swapping in the quantized layers (floats
  freed after). The **no-float** path avoids that peak: `from_weights(...,
  low_memory=True)` / `quantize_and_load` build an int skeleton and quantize each
  tensor as it streams in, and `load_quantized(skeleton, ..., dummy_inputs=...)`
  reloads a saved quantized artifact the same way. The no-float load needs the
  model's converter to assign through `model.weights` (the standard LLM pattern);
  it verifies every quantized layer was filled and errors clearly otherwise, so it
  never silently corrupts: fall back to the float path for those models.
- **Coverage.** `Dense`, `EinsumDense`, `Embedding`, and fused-SwiGLU MoE expert
  banks (`gate_up_proj`/`down_proj`) are quantized; other custom weight layouts
  stay float. A `Dense`/`Embedding` stored inside a Python list (rare:
  kerasformers uses attributes) is skipped with a warning. `dequantize_model`
  reverts `Dense`/`Embedding`; quantized `EinsumDense` / experts stay quantized
  (they still run correctly). Tied-output LLMs that read `token_embedding.embeddings`
  for the logit projection keep working: `QuantizedEmbedding` exposes a
  dequantizing `embeddings` property.
- **Functional models are fully covered**, including Denses nested in custom
  blocks and nested `Functional` sub-models (encoder/decoder): after cloning the
  graph, the in-place swap descends into each block and recurses into sub-models.
  Functional **encoder-decoder ASR** (Whisper / Speech2Text / Moonshine) is the
  exception: it's *partially* quantized (cloneable parts like the encoder), but
  the decoder's weight-capturing `Lambda` lm_head can't be cloned so it stays
  float, and `clone_model` returns a plain `Functional` (dropping cached-
  generation methods), so quantized ASR is forward-only, not for `generate()`.
- **fp8 is torch / jax only.** TensorFlow lacks the float8 casts, so `"fp8"`
  raises a clear error there: use `"int8"` for a tf-portable ~4× option.
- **No calibrated PTQ (GPTQ / AWQ).** This is round-to-nearest weight
  quantization; calibration-based methods for higher int4 accuracy are not
  included.
