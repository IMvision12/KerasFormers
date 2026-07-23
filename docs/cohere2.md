# Cohere 2 (Command R7B)

<div style="background:#fdecea; border:1px solid #f5c6c0; border-radius:3px; padding:12px 16px; color:#4a2626;">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored on the kerasformers
release page. <code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
These are <b>gated</b>: accept the license at <a href="https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024" style="color:#1a5c8a;">CohereLabs/c4ai-command-r7b-12-2024</a>, then authenticate with
<code>huggingface-cli login</code> or <code>export HF_TOKEN=...</code>.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/" style="color:#1a5c8a;">Loading Weights</a>.
</div>
<br>

Cohere's second-generation decoder-only LLM, ported to pure Keras 3. It keeps
Cohere's mean-centered LayerNorm, parallel attention/MLP and `logit_scale`, and
adds alternating sliding-window and global attention layers
(`sliding_window_pattern`).

Links:

- HF docs: [transformers/model_doc/cohere2](https://huggingface.co/docs/transformers/model_doc/cohere2)

See also [cohere.md](cohere.md), [cohere2_moe.md](cohere2_moe.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `c4ai-command-r7b-12-2024` | [`CohereLabs/c4ai-command-r7b-12-2024`](https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024) |
| `c4ai-command-r7b-arabic-02-2025` | [`CohereLabs/c4ai-command-r7b-arabic-02-2025`](https://huggingface.co/CohereLabs/c4ai-command-r7b-arabic-02-2025) |
| `command-a-03-2025` | [`CohereLabs/c4ai-command-a-03-2025`](https://huggingface.co/CohereLabs/c4ai-command-a-03-2025) |
| `command-a-reasoning-08-2025` | [`CohereLabs/command-a-reasoning-08-2025`](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025) |
| `command-a-translate-08-2025` | [`CohereLabs/command-a-translate-08-2025`](https://huggingface.co/CohereLabs/command-a-translate-08-2025) |
| `tiny-aya-base` | [`CohereLabs/tiny-aya-base`](https://huggingface.co/CohereLabs/tiny-aya-base) |
| `tiny-aya-earth` | [`CohereLabs/tiny-aya-earth`](https://huggingface.co/CohereLabs/tiny-aya-earth) |
| `tiny-aya-global` | [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) |
| `tiny-aya-water` | [`CohereLabs/tiny-aya-water`](https://huggingface.co/CohereLabs/tiny-aya-water) |

## API

### `Cohere2Model`

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
| `sliding_window` | `4096` | local attention span |
| `sliding_window_pattern` | `4` | one global layer every N |
| `norm_eps` | `1e-05` | RMSNorm epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |
| `attention_bias` | `False` | add bias terms to the qkv projections |
| `logit_scale` | `0.0625` | output logit scaling |
| `tie_embeddings` | `True` | reuse the embedding matrix as the LM head |
| `layer_types` | `None` |  |

### `Cohere2Generate`

`Cohere2Model` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Cohere2Model`.

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

### `Cohere2Tokenizer`

Tokenizer on the `tokenizers` backend.

```python
Cohere2Tokenizer(hf_id=None, tokenizer_file=None)
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

from kerasformers.models.cohere2 import Cohere2Generate, Cohere2Tokenizer

model = Cohere2Generate.from_weights("c4ai-command-r7b-12-2024")
tokenizer = Cohere2Tokenizer.from_weights("c4ai-command-r7b-12-2024")

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
from kerasformers.models.cohere2 import Cohere2Model

backbone = Cohere2Model.from_weights("c4ai-command-r7b-12-2024")
hidden = backbone(inputs)["last_hidden_state"]   # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Cohere2Generate.from_weights("hf:CohereLabs/c4ai-command-r7b-12-2024")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Cohere2Generate.from_weights(
    "c4ai-command-r7b-12-2024", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
