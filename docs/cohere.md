# Cohere (Command-R)

Cohere's Command-R decoder-only LLM, ported to pure Keras 3. Three things set it
apart from the Llama-style block: LayerNorm is **mean-centered** rather than
RMSNorm, attention and the MLP run **in parallel** off the same normed input, and
the output logits are scaled by `logit_scale`.

Links:

- HF docs: [transformers/model_doc/cohere](https://huggingface.co/docs/transformers/model_doc/cohere)

See also [cohere2.md](cohere2.md), [cohere2_moe.md](cohere2_moe.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `c4ai-command-r-v01` | [`CohereForAI/c4ai-command-r-v01`](https://huggingface.co/CohereForAI/c4ai-command-r-v01) |
| `c4ai-command-r-08-2024` | [`CohereForAI/c4ai-command-r-08-2024`](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024) |
| `c4ai-command-r-plus` | [`CohereForAI/c4ai-command-r-plus`](https://huggingface.co/CohereForAI/c4ai-command-r-plus) |
| `c4ai-command-r-plus-08-2024` | [`CohereForAI/c4ai-command-r-plus-08-2024`](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024) |
| `aya-23-8B` | [`CohereForAI/aya-23-8B`](https://huggingface.co/CohereForAI/aya-23-8B) |
| `aya-23-35B` | [`CohereForAI/aya-23-35B`](https://huggingface.co/CohereForAI/aya-23-35B) |
| `aya-expanse-8b` | [`CohereForAI/aya-expanse-8b`](https://huggingface.co/CohereForAI/aya-expanse-8b) |
| `aya-expanse-32b` | [`CohereForAI/aya-expanse-32b`](https://huggingface.co/CohereForAI/aya-expanse-32b) |

## API

### `CohereModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `256000` | token vocabulary size |
| `embed_dim` | `8192` | model width |
| `num_layers` | `40` | decoder blocks |
| `num_heads` | `64` | query heads |
| `num_kv_heads` | `64` | key/value heads (GQA) |
| `head_dim` | `None` | per-head width |
| `mlp_dim` | `22528` | MLP inner width |
| `use_qk_norm` | `False` | RMS-normalize queries and keys before attention |
| `norm_eps` | `1e-05` | RMSNorm epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |
| `attention_bias` | `False` | add bias terms to the qkv projections |
| `logit_scale` | `0.0625` | output logit scaling |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |

### `CohereGenerate`

`CohereModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `CohereModel`.

```python
generate(input_ids, attention_mask=None, max_new_tokens=None,
         eos_token_id=None, sampler=None, seed=None, **prefill_inputs)
```

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | `(batch, seq)` token ids |
| `attention_mask` | `None` | `(batch, seq)` 1 = keep, 0 = padding |
| `max_new_tokens` | `None` | tokens to generate |
| `eos_token_id` | `None` | stop token (defaults to the tokenizer's) |
| `sampler` | `None` | sampling strategy; greedy when unset |
| `seed` | `None` | seed for stochastic samplers |

### `CohereTokenizer`

Tokenizer on the `tokenizers` backend.

```python
CohereTokenizer(hf_id=None, tokenizer_file=None)
```

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull the tokenizer files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It
accepts a plain string or a list of strings (a batch). This tokenizer has no chat
template, so pass a prompt you have already rendered rather than a message list.
Decode with `.decode(ids)` for one sequence or `.batch_decode(ids)` for a batch.

## End-to-end example

### Single input

```python
import os
os.environ["KERAS_BACKEND"] = "torch"   # or "jax" / "tensorflow"

from kerasformers.models.cohere import CohereGenerate, CohereTokenizer

model = CohereGenerate.from_weights("c4ai-command-r-v01")
tokenizer = CohereTokenizer.from_weights("c4ai-command-r-v01")

inputs = tokenizer("Explain rotary embeddings in one sentence.")
outputs = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(outputs[0]))
```

### Batch

Pass a list of strings. The tokenizer pads them and `generate` runs the batch
together:

```python
prompts = [
    "The capital of France is",
    "In one sentence, what is a transformer?",
    "Write a haiku about GPUs.",
]
inputs = tokenizer(prompts)              # {"input_ids": (3, seq), "attention_mask": (3, seq)}
outputs = model.generate(**inputs, max_new_tokens=64)

for text in tokenizer.batch_decode(outputs):
    print(text)
```

### Backbone only

```python
from kerasformers.models.cohere import CohereModel

backbone = CohereModel.from_weights("c4ai-command-r-v01")
hidden = backbone(inputs)["last_hidden_state"]   # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = CohereGenerate.from_weights("hf:CohereForAI/c4ai-command-r-v01")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = CohereGenerate.from_weights(
    "c4ai-command-r-v01", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
