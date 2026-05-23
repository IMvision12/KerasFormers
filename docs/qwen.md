# Qwen text LLMs (Qwen2 · Qwen3 · Qwen3.5)

**Papers**:
[Qwen2](https://arxiv.org/abs/2407.10671) ·
[Qwen3](https://arxiv.org/abs/2505.09388) ·
[Qwen3.5 / Qwen3-Next](https://arxiv.org/abs/2505.09388)

The pure-text decoder-only LLMs behind Alibaba's Qwen series, as **pure Keras 3**
ports with bit-close parity to HuggingFace (verified on real checkpoints — see
below). Each lives in its own self-contained folder (`qwen2/`, `qwen3/`,
`qwen3_5/`) and runs unmodified on TensorFlow / Torch / JAX.

- **Qwen2** — dense decoder: RMSNorm, GQA with **q/k/v bias**, 1-D RoPE, SwiGLU.
- **Qwen3** — like Qwen2 but **per-head QK-norm** and **no qkv bias**.
- **Qwen3.5** — the **Qwen3-Next hybrid**: most layers are **Gated-DeltaNet**
  linear attention (depthwise causal conv1d + delta-rule recurrence + gated
  RMSNorm); every 4th layer is **gated full attention** (QK-norm, partial rotary,
  sigmoid output gate). Norms are zero-centered `(1 + weight)`.

## On-the-fly weight loading

Each family exposes two classes (mirroring HF's `*Model` / `*ForCausalLM`):

- **`*Model`** — base decoder; its `call` returns features (`last_hidden_state`).
- **`*Generate`** — adds the LM head + greedy `.generate()`; `call` returns
  `logits`.

These are **not** uploaded as kerasformers release weights; they convert on the
fly from the Hugging Face checkpoints (safetensors downloaded and mapped at load
time):

```python
from kerasformers.models.qwen2 import Qwen2Generate, Qwen2Model
from kerasformers.models.qwen3 import Qwen3Generate
from kerasformers.models.qwen3_5 import Qwen3_5Generate

gen = Qwen2Generate.from_weights("hf:Qwen/Qwen2-0.5B")     # text generation
feats = Qwen2Model.from_weights("hf:Qwen/Qwen2-0.5B")      # features
gen = Qwen3Generate.from_weights("hf:Qwen/Qwen3-0.6B")
gen = Qwen3_5Generate.from_weights("hf:Qwen/Qwen3.5-0.8B")
```

bf16 checkpoints are cast to float32 on transfer; tied and untied LM heads are
both handled.

> **Qwen3.5 is a multimodal series.** The released checkpoints
> (`Qwen3_5ForConditionalGeneration`) bundle a vision tower; this port is the
> **text backbone** (`model_type` `qwen3_5` / `qwen3_5_text`), loaded from the
> checkpoint's `model.language_model.*` tensors (vision and the MTP head are
> ignored). For pure text the three M-RoPE position axes coincide, so rotary
> reduces to standard 1-D partial rope.

## Verified parity

Each port was validated against the HF reference (eager, greedy) on a real text
forward pass — **argmax agreement 1.0000** at every position, and **token-exact**
greedy generation:

| Model | Checkpoint | max \|Δ logits\| | argmax match | greedy |
|---|---|---|---|---|
| Qwen2 | `Qwen/Qwen2-0.5B` | 3.1e-5 | 1.0000 | token-exact |
| Qwen3 | `Qwen/Qwen3-0.6B` | 2.2e-5 | 1.0000 | token-exact |
| Qwen3.5 | `Qwen/Qwen3.5-0.8B` | 1.5e-5 | 1.0000 | token-exact |

(The Qwen3.5 residual is chunked-vs-recurrent Gated-DeltaNet fp accumulation —
the kernels are algebraically identical.)

## Forward pass

Both classes take the same input dict. `*Model` returns features; `*Generate`
adds logits:

```python
inputs = {"input_ids": input_ids}          # (B, L) int token ids
feats["last_hidden_state"]                 # Qwen3Model    -> (B, L, hidden_size)
gen(inputs)["logits"]                      # Qwen3Generate -> (B, L, vocab_size)
```

These are token-id models — no spatial axes, so channels_first/last does not
apply (handled like the audio models).

## Generation

`.generate()` does greedy decoding with a KV cache (Qwen3.5 additionally carries
the per-layer conv state + delta-rule recurrent state for its linear layers):

```python
from kerasformers.models.qwen3 import Qwen3Generate, Qwen3Tokenizer

model = Qwen3Generate.from_weights("hf:Qwen/Qwen3-0.6B")
tok = Qwen3Tokenizer()
prompt = tok.apply_chat_template([{"role": "user", "content": "Name three primes."}])
ids = tok(prompt)["input_ids"]
out = model.generate(ids, max_new_tokens=128, eos_token_id=(tok.eos_token_id,))
print(tok.decode(out[0]))
```

## What differs between the families

| | Qwen2 | Qwen3 | Qwen3.5 |
|---|---|---|---|
| Token mixer | GQA attention | GQA attention | **hybrid** linear / full |
| Linear attention | — | — | **Gated-DeltaNet** (conv1d + delta rule) |
| QK-norm | no | **yes** | yes (full layers) |
| QKV bias | **yes** | no | no |
| RoPE | 1-D full | 1-D full | **partial** (factor 0.25) |
| Norm | RMSNorm | RMSNorm | **zero-centered** `(1+w)` + gated |
| Output gate | — | — | **sigmoid gate** (full attention) |

Each folder defines its own layer classes (no cross-family imports); the
weight conversion (`convert_*_hf_to_keras.py`) maps HF safetensors to Keras.

## Citation

```bibtex
@article{Qwen2,
  title={Qwen2 Technical Report},
  author={Yang, An and others}, journal={arXiv:2407.10671}, year={2024}
}
@article{Qwen3,
  title={Qwen3 Technical Report},
  author={Yang, An and others}, journal={arXiv:2505.09388}, year={2025}
}
```
