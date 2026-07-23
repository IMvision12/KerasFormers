# Qwen3.5-MoE

<div style="background:#fdecea; border:1px solid #f5c6c0; border-radius:3px; padding:12px 16px; color:#4a2626;">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored on the kerasformers
release page. <code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/" style="color:#1a5c8a;">Loading Weights</a>.
</div>
<br>

The Mixture-of-Experts variant of Qwen3.5, ported to pure Keras 3. It keeps the
Gated-DeltaNet / full-attention hybrid and replaces the MLP with a routed expert
bank.

Memory is governed by **total** parameters, not active ones.


See also [qwen3_5.md](qwen3_5.md), [qwen3_moe.md](qwen3_moe.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `qwen3-next-80b-a3b-instruct` | [`Qwen/Qwen3-Next-80B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) |
| `qwen3-next-80b-a3b-thinking` | [`Qwen/Qwen3-Next-80B-A3B-Thinking`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) |

## API

### `Qwen35MoeModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `2048` | model width |
| `mlp_dim` | `5120` | MLP inner width |
| `num_layers` | `48` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `head_dim` | `256` | per-head width |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `rope_theta` | `10000000.0` | rotary base frequency |
| `partial_rotary_factor` | `0.25` | fraction of each head that gets rotated |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |
| `full_attention_interval` | `4` |  |
| `linear_conv_kernel_dim` | `4` |  |
| `linear_key_head_dim` | `128` |  |
| `linear_value_head_dim` | `128` |  |
| `linear_num_key_heads` | `16` |  |
| `linear_num_value_heads` | `32` |  |
| `num_experts` | `512` | expert count |
| `num_experts_per_tok` | `10` | experts routed per token |
| `moe_mlp_dim` | `512` | per-expert inner width |
| `shared_mlp_dim` | `512` |  |
| `norm_topk_prob` | `True` |  |
| `decoder_sparse_step` | `1` |  |
| `mlp_only_layers` | `()` |  |

### `Qwen35MoeGenerate`

`Qwen35MoeModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Qwen35MoeModel`.

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

### `Qwen35MoeTokenizer`

Tokenizer on the `tokenizers` backend.

```python
Qwen35MoeTokenizer(hf_id=None, tokenizer_file=None)
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

from kerasformers.models.qwen3_5_moe import Qwen35MoeGenerate, Qwen35MoeTokenizer

model = Qwen35MoeGenerate.from_weights("qwen3-next-80b-a3b-instruct")
tokenizer = Qwen35MoeTokenizer.from_weights("qwen3-next-80b-a3b-instruct")

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
from kerasformers.models.qwen3_5_moe import Qwen35MoeModel

backbone = Qwen35MoeModel.from_weights("qwen3-next-80b-a3b-instruct")
hidden = backbone(inputs)["last_hidden_state"]   # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Qwen35MoeGenerate.from_weights("hf:Qwen/Qwen3-Next-80B-A3B-Instruct")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen35MoeGenerate.from_weights(
    "qwen3-next-80b-a3b-instruct", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
