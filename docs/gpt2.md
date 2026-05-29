# GPT-2 (language model)

OpenAI's GPT-2 in **pure Keras 3** — the classic decoder-only language model:
learned token + absolute-position embeddings, pre-LayerNorm causal transformer
blocks, a final LayerNorm, and a tied LM head. One implementation runs unmodified
on **TensorFlow / Torch / JAX**, with bit-close parity to Hugging Face. Weights
are **converted on the fly** from the Hugging Face repos (nothing re-hosted).

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

## Loading (on the fly, no release weights)

```python
from kerasformers.models.gpt2 import GPT2Generate, GPT2Tokenizer

model = GPT2Generate.from_weights("gpt2")        # or "gpt2-medium" / "-large" / "-xl"
tok = GPT2Tokenizer()
ids = model.generate(tok("The meaning of life is")["input_ids"], max_new_tokens=40)
print(tok.decode(ids[0]))
```

### Available variants

| Variant | layers | embed_dim | heads |
|---|---|---|---|
| `gpt2` | 12 | 768 | 12 |
| `gpt2-medium` | 24 | 1024 | 16 |
| `gpt2-large` | 36 | 1280 | 20 |
| `gpt2-xl` | 48 | 1600 | 25 |

## Verified parity

`GPT2Generate` logits vs the real `openai-community/gpt2` (HF, eager attention):
**max |Δ| 4.6e-5**, argmax 100% agree. Build + forward + `.generate()` pass on
TF / Torch / JAX.
