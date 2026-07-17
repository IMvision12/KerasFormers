# Qwen3

Alibaba's Qwen3 dense decoder-only LLM, ported to pure Keras 3. It drops Qwen2's
QKV biases and adds QK-RMSNorm (queries and keys are normalized before
attention), keeping RMSNorm, SwiGLU and grouped-query attention otherwise.

Links:

- Paper: [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- HF docs: [transformers/model_doc/qwen3](https://huggingface.co/docs/transformers/model_doc/qwen3)

See also [qwen2.md](qwen2.md), [qwen3_moe.md](qwen3_moe.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `qwen3-0.6b` | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) |
| `qwen3-0.6b-base` | [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) |
| `qwen3-1.7b` | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) |
| `qwen3-1.7b-base` | [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base) |
| `qwen3-4b` | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) |
| `qwen3-4b-base` | [`Qwen/Qwen3-4B-Base`](https://huggingface.co/Qwen/Qwen3-4B-Base) |
| `qwen3-8b` | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) |
| `qwen3-8b-base` | [`Qwen/Qwen3-8B-Base`](https://huggingface.co/Qwen/Qwen3-8B-Base) |
| `qwen3-14b` | [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B) |
| `qwen3-14b-base` | [`Qwen/Qwen3-14B-Base`](https://huggingface.co/Qwen/Qwen3-14B-Base) |
| `qwen3-32b` | [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) |

## API

### `Qwen3Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `1024` | model width |
| `mlp_dim` | `3072` | MLP inner width |
| `num_layers` | `28` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |

### `Qwen3Generate`

`Qwen3Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Qwen3Model`.

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

### `Qwen3Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
Qwen3Tokenizer(hf_id=None, tokenizer_file=None)
```

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull the tokenizer files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It
accepts a plain string, a list of strings (a batch), or a chat-message list, which is
routed through `apply_chat_template` automatically. Decode with `.decode(ids)` for one
sequence or `.batch_decode(ids)` for a batch.

## End-to-end example

### Single input

```python
import os
os.environ["KERAS_BACKEND"] = "torch"   # or "jax" / "tensorflow"

from kerasformers.models.qwen3 import Qwen3Generate, Qwen3Tokenizer

model = Qwen3Generate.from_weights("qwen3-0.6b")
tokenizer = Qwen3Tokenizer.from_weights("qwen3-0.6b")

inputs = tokenizer([{"role": "user", "content": "Explain rotary embeddings in one sentence."}])
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
from kerasformers.models.qwen3 import Qwen3Model

backbone = Qwen3Model.from_weights("qwen3-0.6b")
hidden = backbone(inputs)["last_hidden_state"]   # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Qwen3Generate.from_weights("hf:Qwen/Qwen3-0.6B")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen3Generate.from_weights(
    "qwen3-0.6b", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
