# Qwen3.5 (Qwen3-Next)

<div style="background:#fdecea; border:1px solid #f5c6c0; border-radius:3px; padding:12px 16px; color:#4a2626;">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored on the kerasformers
release page. <code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/" style="color:#1a5c8a;">Loading Weights</a>.
</div>
<br>

Alibaba's Qwen3.5 hybrid-attention LLM, ported to pure Keras 3. It interleaves
Gated-DeltaNet linear-attention layers with periodic full-attention layers
(`full_attention_interval`), keeping partial rotary embeddings and QK-RMSNorm on
the full-attention path.


See also [qwen3.md](qwen3.md), [qwen3_5_moe.md](qwen3_5_moe.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `qwen3.5-0.8b` | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| `qwen3.5-0.8b-base` | [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) |
| `qwen3.5-2b` | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) |
| `qwen3.5-2b-base` | [`Qwen/Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base) |
| `qwen3.5-4b` | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) |
| `qwen3.5-4b-base` | [`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base) |
| `qwen3.5-9b` | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) |
| `qwen3.5-9b-base` | [`Qwen/Qwen3.5-9B-Base`](https://huggingface.co/Qwen/Qwen3.5-9B-Base) |
| `qwen3.5-27b` | [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) |

## API

### `Qwen3_5Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `248320` | token vocabulary size |
| `embed_dim` | `1024` | model width |
| `mlp_dim` | `3584` | MLP inner width |
| `num_layers` | `24` | decoder blocks |
| `num_heads` | `8` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `head_dim` | `256` | per-head width |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `rope_theta` | `10000000.0` | rotary base frequency |
| `partial_rotary_factor` | `0.25` | fraction of each head that gets rotated |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |
| `full_attention_interval` | `4` |  |
| `linear_conv_kernel_dim` | `4` |  |
| `linear_key_head_dim` | `128` |  |
| `linear_value_head_dim` | `128` |  |
| `linear_num_key_heads` | `16` |  |
| `linear_num_value_heads` | `16` |  |

### `Qwen3_5Generate`

`Qwen3_5Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Qwen3_5Model`.

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

### `Qwen3_5Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
Qwen3_5Tokenizer(hf_id=None, tokenizer_file=None)
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

from kerasformers.models.qwen3_5 import Qwen3_5Generate, Qwen3_5Tokenizer

model = Qwen3_5Generate.from_weights("qwen3.5-0.8b")
tokenizer = Qwen3_5Tokenizer.from_weights("qwen3.5-0.8b")

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
from kerasformers.models.qwen3_5 import Qwen3_5Model

backbone = Qwen3_5Model.from_weights("qwen3.5-0.8b")
hidden = backbone(inputs)["last_hidden_state"]   # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Qwen3_5Generate.from_weights("hf:Qwen/Qwen3.5-0.8B")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen3_5Generate.from_weights(
    "qwen3.5-0.8b", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
