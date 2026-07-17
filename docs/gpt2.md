# GPT-2 (language model)

OpenAI's GPT-2 in **pure Keras 3**, the classic decoder-only language model:
learned token + absolute-position embeddings, pre-LayerNorm causal transformer
blocks, a final LayerNorm, and a tied LM head. One implementation runs unmodified
on **TensorFlow / Torch / JAX**, with bit-close parity to Hugging Face. Weights
load from the kerasformers GitHub release: `gpt2_large` / `gpt2_xl` are **sharded**
(`.weights.json` index + shards, since they exceed GitHub's 2 GB asset cap).

**Paper**: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

| Class | Module | Output |
|---|---|---|
| `GPT2Model` | `kerasformers.models.gpt2` | `{"last_hidden_state": (B, L, embed_dim)}` |
| `GPT2Generate` | `kerasformers.models.gpt2` | `{"logits": (B, L, vocab), "last_hidden_state": ...}` + `.generate()` |
| `GPT2Tokenizer` | `kerasformers.models.gpt2` | byte-level BPE → `input_ids` / `attention_mask` |

`GPT2Model` is a subclassed (imperative) `SubclassedBaseModel`; `GPT2Generate`
adds the tied LM head (the transposed token embedding) and greedy `.generate()`
with a KV cache. The attention/MLP projections use GPT-2's `Conv1D` `(in, out)`
weight layout (copied without transposing), the MLP activation is the tanh-`gelu`
approximation (`gelu_new`), and the blocks are pre-LayerNorm with a final `ln_f`.

## Loading

```python
from kerasformers.models.gpt2 import GPT2Generate, GPT2Tokenizer

model = GPT2Generate.from_weights("gpt2")        # or "gpt2_medium" / "_large" / "_xl"
tok = GPT2Tokenizer()
ids = model.generate(tok("The meaning of life is")["input_ids"], max_new_tokens=40)
print(tok.decode(ids[0]))
```

`from_weights("hf:openai-community/gpt2")` also works (on-the-fly conversion).
The release `.weights.h5` / sharded `.weights.json` are produced with
`KERAS_BACKEND=torch python -m kerasformers.models.gpt2.convert_gpt2_hf_to_keras`
(large/xl are saved with `max_shard_size`).

### Available variants

| Variant | layers | embed_dim | heads |
|---|---|---|---|
| `gpt2` | 12 | 768 | 12 |
| `gpt2_medium` | 24 | 1024 | 16 |
| `gpt2_large` | 36 | 1280 | 20 |
| `gpt2_xl` | 48 | 1600 | 25 |

## Verified parity

`GPT2Generate` logits vs the real `openai-community/gpt2` (HF, eager attention):
**max |Δ| 4.6e-5**, argmax 100% agree. Build + forward + `.generate()` pass on
TF / Torch / JAX.
