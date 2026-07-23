# Qwen3-MoE

<div style="background:#fdecea; border:1px solid #f5c6c0; border-radius:3px; padding:12px 16px; color:#4a2626;">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored on the kerasformers
release page. <code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/" style="color:#1a5c8a;">Loading Weights</a>.
</div>
<br>

The Mixture-of-Experts variant of Qwen3, ported to pure Keras 3. It keeps Qwen3's
QK-RMSNorm attention and bias-free projections, replacing each MLP with a
softmax-routed expert bank.

Memory is governed by **total** parameters, not active ones.

Links:

- Paper: [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- HF docs: [transformers/model_doc/qwen3_moe](https://huggingface.co/docs/transformers/model_doc/qwen3_moe)

See also [qwen3.md](qwen3.md), [qwen2_moe.md](qwen2_moe.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `qwen3-30b-a3b` | [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| `qwen3-30b-a3b-instruct-2507` | [`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) |
| `qwen3-235b-a22b` | [`Qwen/Qwen3-235B-A22B`](https://huggingface.co/Qwen/Qwen3-235B-A22B) |

## API

### `Qwen3MoeModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `2048` | model width |
| `num_layers` | `48` | decoder blocks |
| `num_heads` | `32` | query heads |
| `num_kv_heads` | `4` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `mlp_dim` | `6144` | MLP inner width |
| `num_experts` | `128` | expert count |
| `num_experts_per_tok` | `8` | experts routed per token |
| `moe_mlp_dim` | `768` | per-expert inner width |
| `norm_topk_prob` | `True` |  |
| `decoder_sparse_step` | `1` |  |
| `mlp_only_layers` | `()` |  |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `norm_eps` | `1e-06` | RMSNorm epsilon |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `Qwen3MoeGenerate`

`Qwen3MoeModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Qwen3MoeModel`.

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

### `Qwen3MoeTokenizer`

Tokenizer on the `tokenizers` backend.

```python
Qwen3MoeTokenizer(hf_id=None, tokenizer_file=None)
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

from kerasformers.models.qwen3_moe import Qwen3MoeGenerate, Qwen3MoeTokenizer

model = Qwen3MoeGenerate.from_weights("qwen3-30b-a3b")
tokenizer = Qwen3MoeTokenizer.from_weights("qwen3-30b-a3b")

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
from kerasformers.models.qwen3_moe import Qwen3MoeModel

backbone = Qwen3MoeModel.from_weights("qwen3-30b-a3b")
hidden = backbone(inputs)["last_hidden_state"]   # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Qwen3MoeGenerate.from_weights("hf:Qwen/Qwen3-30B-A3B")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen3MoeGenerate.from_weights(
    "qwen3-30b-a3b", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
