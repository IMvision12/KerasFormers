# Quantization (int8 / int4 / fp8 / mxfp4)

kerasformers ships its **own** weight-only int8 / int4 / fp8 / mxfp4 quantization in
`kerasformers/quantization/`: a from-scratch, backend-agnostic implementation
(pure `keras.ops`), not Keras's built-in `model.quantize`. It shrinks a model
~4× (int8 / fp8) or ~8× (int4 / mxfp4) so larger checkpoints fit in memory. int8 /
int4 / mxfp4 run on TensorFlow / Torch / JAX; fp8 (float8-e4m3) is torch / jax only.

## Choosing a scheme

| Scheme | Size | Cosine | Backends | Pick it when |
|---|---|---|---|---|
| [**int8**](quantization_int8.md) | ~3.8× smaller | ~0.9999 | all three | The default. Near-free accuracy, use it whenever it fits. |
| [**int4**](quantization_int4.md) | ~5.8–8× smaller | ~0.98 | all three | int8 does not fit. Block-wise, `group_size` is the knob. |
| [**fp8**](quantization_fp8.md) | ~3.8× smaller | ~0.9994 | torch / jax | Same size as int8, better on heavy-tailed weights. Measure both. |
| [**mxfp4**](quantization_mxfp4.md) | ~8× smaller | ~0.98 | all three | 4-bit *float* (OCP e2m1). The format GPT-OSS ships; contracting dims must be multiples of 32. |

Each page covers that scheme's math, storage layout, measured accuracy, and its own config
class. The rest of this page is the machinery they share.

> **[MXFP4](quantization_mxfp4.md)** does double duty: it is both a general
> `quantize_model(model, "mxfp4")` scheme (applied to any model whose kernels have
> 32-multiple contracting dims) **and** the native on-disk format of GPT-OSS, whose
> experts are shipped packed and dequantized on the fly with the same primitives.

## Quick start

```python
from kerasformers.models.qwen3 import Qwen3Generate

# load + quantize in one call
model = Qwen3Generate.from_weights("qwen3-4b", quantization="int8")  # ~4x smaller
model = Qwen3Generate.from_weights("qwen3-4b", quantization="int4")  # ~8x smaller
model = Qwen3Generate.from_weights("qwen3-4b", quantization="fp8")  # ~4x (torch/jax)

# or quantize a model you already built/loaded
from kerasformers.quantization import quantize_model

quantize_model(model, "int8")  # in place
quantize_model(model, "int4", group_size=64)  # int4 block size (default 32)
quantize_model(model, "fp8")  # float8 e4m3 (torch / jax)
quantize_model(model, "mxfp4")  # OCP 4-bit float, fixed 32-value blocks
```

`quantization=` is wired through `from_weights` for every model (Hub Keras repos,
bare LLM/VLM variants, **and** `hf:` repos).

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
    mode="int4",
    group_size=128,
    skip_modules=("lm_head",),  # keep these layers in float
    quantize_embeddings=True,
    overrides={"decoder_layer_0": "int8"},  # per-layer precision
)
quantize_model(model, cfg)
quantize_model(model, "int4-g128")  # or a named scheme
```

**Save / load / revert.** A quantized model saves and reloads itself quantized through
ordinary Keras save (the quantization is carried in `get_config` and re-applied in
`from_config`, see [`KfQuantizer`](#loading-a-pre-quantized-repo-kfquantizer)):

```python
import keras
from kerasformers.quantization import dequantize_model, get_kf_quantizer

# Full save: the quantization round-trips automatically.
model.save("model.keras")
model = keras.saving.load_model("model.keras")  # rebuilt quantized, weights loaded

# Weights-only (.weights.h5) carries values, not structure, so the target must already
# be quantized before load_weights. From a Hub repo that is automatic (kf_config's
# quantization_config drives it):
model = Qwen3Generate.from_weights("kerasformers/qwen3-4b-int8")

# Into a hand-built model, apply the quantizer first, then load_weights:
model.save_weights("model.weights.h5")
skeleton = Qwen3Generate.from_weights("qwen3-4b", load_weights=False)
get_kf_quantizer({"quant_method": "int8"}).preprocess_model(skeleton)
skeleton(dummy_inputs)  # build the now-int8 skeleton
skeleton.load_weights("model.weights.h5")

dequantize_model(model)  # revert to float layers
```

**MoE and functional models:** `quantize_model` also quantizes **fused MoE
experts** (the `gate_up_proj` / `down_proj` banks of Qwen/GLM/DeepSeek-MoE, along
the contracted axis) and **functional / vision** models (ViT, CLIP, …). A
functional graph can't be mutated in place, so it is **cloned**: use the
returned model:

```python
qmodel = quantize_model(vit_model, "int8")  # functional -> returns a NEW model
```

## Loading a pre-quantized repo (`KfQuantizer`)

Everything above *applies* quantization to a float model. A repo can also **ship
already quantized** and declare it in `kf_config.json` with a transformers-style
block:

```json
"quantization_config": { "quant_method": "mxfp4" }
```

`from_weights` reads that block and **auto-applies** the matching quantizer, so a
quantized repo loads with **no flag**:

```python
# reads quantization_config -> loads bf16 dense + mxfp4 experts, by default
model = GptOssGenerate.from_weights("kerasformers/gpt-oss-20b")
```

The models stay **quantization-agnostic** (no per-model flags): the model builds the
plain float architecture, and a `KfQuantizer` swaps in the quantized layers **before
the weights load**, exactly like transformers'
`HfQuantizer._process_model_before_weight_loading`. `KfQuantizer` is a **second
level** above the tensor-level `BaseQuantizer`:

| level | class | job |
|---|---|---|
| tensor | `BaseQuantizer` (`Int8Quantizer`, `MXFP4Quantizer`, …) | quantize / dequantize one weight along an axis |
| model | `KfQuantizer` (transformers' `HfQuantizer` analog) | read `quantization_config`, swap modules before load |

`get_kf_quantizer(block)` dispatches on `quant_method`:

- `mxfp4` -> `Mxfp4KfQuantizer` (GPT-OSS native: swaps the float `GptOssExperts` for
  the packed `GptOssMXFP4Experts`).
- `int8` / `int4` / `fp8` -> `WeightOnlyKfQuantizer` (generic: builds the int / fp8
  skeleton via `quantize_skeleton`).

**Save round-trips itself.** The applied quantization is stamped on the model
(`model._quantization_config`) and carried through `get_config` / `from_config`, so a
quantized model **saves and reloads itself quantized** via an ordinary Keras save,
no export step and no re-quantization:

```python
model.save("m.keras")
reloaded = keras.saving.load_model("m.keras")  # rebuilds bf16 + mxfp4 automatically
```

Subclassed models need this hook (Keras rebuilds them from `get_config`, which would
otherwise recreate plain layers); functional models round-trip natively because Keras
serializes each layer, quantized ones included, on its own.

## How it works

Weight-only quantization: the weights are stored quantized and **dequantized on
the fly** inside each layer's `call`, so the matmul still runs in the activation
dtype. No special int kernels are needed, which is why it is fully
backend-agnostic.

- **[int8](quantization_int8.md)**: per-channel symmetric absmax, one float scale
  per output channel over the contracting axis, `scale = max|w| / 127`.
- **[int4](quantization_int4.md)**: block-wise symmetric absmax, `scale = max|w| / 7`
  per block of `group_size`, packed two values per byte.
- **[fp8](quantization_fp8.md)**: per-channel absmax cast into the native
  `float8_e4m3fn` dtype, `scale = max|w| / 448`. torch / jax only.
- **[mxfp4](quantization_mxfp4.md)**: OCP 4-bit float (e2m1) in fixed 32-value
  blocks, each with a shared e8m0 (power-of-two) `uint8` scale; values pack two
  per byte. Same ~8× as int4, but a *float* grid and a `uint8` (not fp32) scale.
- **Embeddings**: int8 with a per-row scale; the lookup gathers int8 rows and
  dequantizes only the gathered slice (for the `int4` and `mxfp4` model modes,
  embeddings stay int8: the 4-bit savings live in the Dense weights).

**N-D kernels.** Quantization is along the **contracting axis**, not a hardcoded
`axis=0`, so the same quantizers serve 2-D `Dense` kernels, N-D `EinsumDense`
kernels (axis derived from the equation: a tuple for int8/fp8, a single axis for
packed int4 / mxfp4), per-row embeddings (`axis=1`), and fused MoE expert banks
(`axis=-1`). Scales keep the reduced axes as size 1 so they broadcast over any
rank with no reshape.

**Robustness.** Scales use an epsilon floor (`max(amax / MAX, ε)`) rather than an
exact-zero test (handles zero and denormal channels), and `dequantize` takes the
compute `dtype`, so `mixed_bfloat16` graphs don't upcast through float32.

`quantize_model` walks the layer tree and **swaps** every built `Dense` →
[`QuantizedDense`](https://github.com/IMvision12/KerasFormers/blob/main/kerasformers/quantization/quantized_layers.py), `EinsumDense`
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
| `BaseQuantizer` | `base/base_quantization.py` | base class (also `kerasformers.base.BaseQuantizer`): `quantize(weight, axis)` / `dequantize(packed, scale, axis, dtype)` / `storage_spec(weight_shape, axis)` + `get_config` / `from_config`; ships `normalize_axes` / `single_axis` |
| `Int8Quantizer` | `int8_quantize.py` | per-channel int8 quantizer (quantize / dequantize methods) |
| `Int4Quantizer` | `int4_quantize.py` | block-wise packed int4 quantizer (any axis via moveaxis; module `effective_group_size`) |
| `Fp8Quantizer` | `fp8_quantize.py` | per-channel float8-e4m3 quantizer (module `fp8_supported`; torch / jax) |
| `MXFP4Quantizer` | `mxfp4_quantize.py` | OCP MXFP4 (e2m1) quantizer, single packed axis, `uint8` e8m0 scale (also `quantize_to_mxfp4` / `dequantize_mxfp4` pack / unpack) |
| `QuantizedDense` / `QuantizedEinsumDense` / `QuantizedEmbedding` / `QuantizedExperts` | `quantized_layers.py` | weight-only drop-in layers (each holds a quantizer); `QuantizedExperts` = fused MoE expert bank, contracting-axis quantized |
| `GptOssMXFP4Experts` | `quantized_layers.py` | GPT-OSS MoE expert bank kept in MXFP4 (native on-disk format), dequantized in `call` (top-k sparse on decode) |
| `QuantizationConfig` / `Int8Config` / `Int4Config` / `Fp8Config` / `Mxfp4Config` / `SCHEMES` | `quant_config.py` | recipe (mode, group_size, skip_modules, quantize_embeddings, overrides) + per-method configs + named presets |
| `quantize_model` / `quantize_functional` | `quantize.py` | in-place (subclassed) / clone (functional) model surgery |
| `quantize_skeleton` / `quantize_and_load` | `quantize.py` | no-float int skeleton / stream a float checkpoint into int storage |
| `KfQuantizer` / `Mxfp4KfQuantizer` / `WeightOnlyKfQuantizer` / `get_kf_quantizer` | `kf_quantizer.py` | model-level quantizers (transformers `HfQuantizer` analog): read a repo's `quantization_config` and swap in packed / int layers before load; dispatched by `quant_method` |
| `dequantize_model` | `quantize.py` | revert quantized layers back to float |

A `QuantizedDense` holds an `Int8Quantizer` / `Int4Quantizer` / `Fp8Quantizer` /
`MXFP4Quantizer` (via `get_quantizer(mode, group_size)`) and uses it for
`storage_spec` (build), `quantize` (from a float `Dense`), and `dequantize` (in
`call`).

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
- **Float path vs no-float path.** By default `quantization=` builds the float
  architecture before swapping in the quantized layers (floats freed after). The
  **no-float** path avoids that peak: `from_weights(..., low_memory=True)` /
  `quantize_and_load` build an int skeleton and quantize each tensor as it streams in.
  The no-float load needs the model's converter to assign through `model.weights` (the
  standard LLM pattern); it verifies every quantized layer was filled and errors
  clearly otherwise, so it never silently corrupts: fall back to the float path for
  those models.
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
