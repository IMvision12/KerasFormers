# GPT (language model)

OpenAI's original GPT (Radford et al. 2018, "openai-gpt") in **pure Keras 3** — a
decoder-only language model with learned token + absolute-position embeddings and
post-LayerNorm causal transformer blocks (no final norm). One implementation runs
unmodified on **TensorFlow / Torch / JAX**, with bit-close parity to Hugging Face.
Weights are **converted on the fly** from the Hugging Face repo.

**Paper**: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

| Class | Module | Output |
|---|---|---|
| `GptModel` | `kerasformers.models.gpt` | `{"last_hidden_state": (B, L, embed_dim)}` |
| `GptGenerate` | `kerasformers.models.gpt` | `{"logits": (B, L, vocab), "last_hidden_state": ...}` + `.generate()` |
| `GptTokenizer` | `kerasformers.models.gpt` | byte-pair encoder → `input_ids` / `attention_mask` |

Same machinery as GPT-2 (`Conv1D` `(in, out)` weights copied without transpose,
`gelu_new` activation, tied LM head) with two differences: the blocks are
**post-LayerNorm** (`ln_1(x + attn(x))`, `ln_2(n + mlp(n))`) and there is **no
final LayerNorm**.

## Loading (on the fly, no release weights)

```python
from kerasformers.models.gpt import GptGenerate, GptTokenizer

model = GptGenerate.from_weights("gpt")          # openai-community/openai-gpt
tok = GptTokenizer()
ids = model.generate(tok("the meaning of life is")["input_ids"], max_new_tokens=40)
print(tok.decode(ids[0]))
```

### Available variant

| Variant | layers | embed_dim | heads | vocab |
|---|---|---|---|---|
| `gpt` | 12 | 768 | 12 | 40478 |

## Verified parity

`GptGenerate` logits vs the real `openai-community/openai-gpt` (HF):
**max |Δ| 1.2e-5**, argmax 100% agree. Build + forward + `.generate()` pass on
TF / Torch / JAX.
