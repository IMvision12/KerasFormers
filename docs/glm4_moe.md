# GLM-4.5 (GLM-4 MoE)

Zhipu's GLM-4.5 / GLM-4.6 Mixture-of-Experts LLM, ported to pure Keras 3. It
pairs partial rotary attention with a DeepSeek-style MoE: fine-grained routed
experts plus shared experts, node-limited routing (`n_group` / `topk_group`) and
the first `first_k_dense` layers left dense.

Rope trap: unlike GLM and GLM-4, which use *interleaved* partial rope, the MoE
models use the NeoX (half-split) layout. Mixing the two silently destroys parity.

Memory is governed by **total** parameters, not active ones: every expert stays
resident.

Links:

- Paper: [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models (arXiv:2508.06471)](https://arxiv.org/abs/2508.06471)
- HF docs: [transformers/model_doc/glm4_moe](https://huggingface.co/docs/transformers/model_doc/glm4_moe)

See also [glm4.md](glm4.md), [glm5_moe.md](glm5_moe.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `glm-4.5` | [`zai-org/GLM-4.5`](https://huggingface.co/zai-org/GLM-4.5) |
| `glm-4.5-air` | [`zai-org/GLM-4.5-Air`](https://huggingface.co/zai-org/GLM-4.5-Air) |
| `glm-4.6` | [`zai-org/GLM-4.6`](https://huggingface.co/zai-org/GLM-4.6) |

## API

### `Glm4MoeModel`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151552` | token vocabulary size |
| `embed_dim` | `4096` | model width |
| `num_layers` | `46` | decoder blocks |
| `num_heads` | `96` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `mlp_dim` | `10944` | MLP inner width |
| `moe_mlp_dim` | `1408` | per-expert inner width |
| `num_experts` | `128` | expert count |
| `num_experts_per_tok` | `8` | experts routed per token |
| `n_shared_experts` | `1` |  |
| `n_group` | `1` |  |
| `topk_group` | `1` |  |
| `norm_topk_prob` | `True` |  |
| `routed_scaling_factor` | `1.0` |  |
| `first_k_dense` | `1` |  |
| `partial_rotary_factor` | `0.5` | fraction of each head that gets rotated |
| `use_qk_norm` | `False` | RMS-normalize queries and keys before attention |
| `norm_eps` | `1e-05` | RMSNorm epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |
| `attention_bias` | `False` | add bias terms to the qkv projections |
| `tie_embeddings` | `False` | reuse the embedding matrix as the LM head |

### `Glm4MoeGenerate`

`Glm4MoeModel` plus a (tied) LM head. Returns `{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. Same constructor
arguments as `Glm4MoeModel`.

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

### `Glm4MoeTokenizer`

Tokenizer on the `tokenizers` backend.

```python
Glm4MoeTokenizer(hf_id=None, tokenizer_file=None)
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

from kerasformers.models.glm4_moe import Glm4MoeGenerate, Glm4MoeTokenizer

model = Glm4MoeGenerate.from_weights("glm-4.5")
tokenizer = Glm4MoeTokenizer.from_weights("glm-4.5")

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
from kerasformers.models.glm4_moe import Glm4MoeModel

backbone = Glm4MoeModel.from_weights("glm-4.5")
hidden = backbone(inputs)["last_hidden_state"]   # (batch, seq, embed_dim)
```

### Loading from the Hub

Any Hub repo with this architecture works via the `hf:` prefix, including
community fine-tunes:

```python
model = Glm4MoeGenerate.from_weights("hf:zai-org/GLM-4.5")
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Glm4MoeGenerate.from_weights(
    "glm-4.5", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
