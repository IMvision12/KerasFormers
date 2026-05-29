# ModernBERT (text encoder)

Answer.AI / LightOn's ModernBERT in **pure Keras 3** — a modernized
bidirectional transformer text encoder (rotary embeddings, alternating
local/global attention, GeGLU feed-forwards, pre-LayerNorm) with its masked-LM,
classification, token-classification, QA, and multiple-choice heads. One
implementation runs unmodified on **TensorFlow / Torch / JAX**, with bit-close
parity to Hugging Face on real checkpoints (see below).

**Paper**: [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663)

| Class | Module | Output |
|---|---|---|
| `ModernBertModel` | `kerasformers.models.modernbert` | `{"last_hidden_state": (B, L, embed_dim)}` |
| `ModernBertMaskedLM` | `kerasformers.models.modernbert` | MLM logits `(B, L, vocab_size)` |
| `ModernBertSequenceClassify` | `kerasformers.models.modernbert` | sequence logits `(B, num_classes)` |
| `ModernBertTokenClassify` | `kerasformers.models.modernbert` | per-token logits `(B, L, num_classes)` |
| `ModernBertQnA` | `kerasformers.models.modernbert` | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `ModernBertMultipleChoice` | `kerasformers.models.modernbert` | per-choice logits `(B, num_choices)` |
| `ModernBertTokenizer` | `kerasformers.models.modernbert` | byte-level BPE → `input_ids` / `attention_mask` |

All models are functional `BaseModel`s; the head classes compose a
`ModernBertModel` backbone. ModernBERT has **no pooler and no token-type
embeddings** — position is injected by rotary embeddings, so the tokenizer emits
only `input_ids` / `attention_mask`. The MLM head is part of the pretrained
checkpoint and loads real weights; the other task heads are randomly initialized
for the official release (ready for fine-tuning) and load trained weights from a
`hf:` fine-tune. The architecture is identical across variants — only the depth
and width differ.

## Architecture notes

- **Rotary position embeddings** with two bases: global layers use
  `global_rope_theta=160000`, local layers use `local_rope_theta=10000`.
- **Alternating attention**: every `global_attn_every_n_layers` (3rd) layer uses
  full attention; the rest use a sliding window of `local_attention` (128) tokens.
- **GeGLU** feed-forward (`Wi` projects to `2 * mlp_dim`, gated, then `Wo`).
- **Pre-LayerNorm** residuals; bias-free linears and LayerNorms; the first
  layer's attention LayerNorm is the identity (the embeddings are already
  normalized).
- The MLM decoder is tied to the token embeddings.

## Loading

Two paths, both via `from_weights`:

- **Official release variant** — `from_weights("modernbert_base")` downloads the
  kerasformers-release `.weights.h5`.
- **`hf:` community fine-tune** — `from_weights("hf:org/repo")` reads the repo's
  `config.json` (architecture + `num_labels`) and loads the checkpoint, including
  the fine-tuned classifier head.

```python
from kerasformers.models.modernbert import ModernBertModel, ModernBertTokenizer

model = ModernBertModel.from_weights("modernbert_base")
tokenizer = ModernBertTokenizer.from_weights("modernbert_base")

out = model(tokenizer("Hello, world."))
out["last_hidden_state"]   # (1, L, 768)
```

### Available variants

| Variant | layers | embed_dim | heads | mlp_dim |
|---|---|---|---|---|
| `modernbert_base` | 22 | 768 | 12 | 1152 |
| `modernbert_large` | 28 | 1024 | 16 | 2624 |

Both variants share one tokenizer (`ModernBertTokenizer.from_weights(variant)`).

## Verified parity

Validated against the Hugging Face reference (eager attention) on a real forward
pass with a sequence long enough to exercise the sliding-window (local)
attention. The larger max residual on `large` is fp32 op-order accumulation over
the deeper/wider stack (mean residual is ~6e-6 and cosine is ~1.0), not an
architectural difference — so the converter gates on cosine (≥ 0.9999), like the
deep DeBERTa-v2 models.

| Model | Checkpoint | max \|Δ\| | cosine |
|---|---|---|---|
| `ModernBertModel` | `answerdotai/ModernBERT-base` | 2.0e-4 | ~1.0 |
| `ModernBertMaskedLM` | `answerdotai/ModernBERT-base` | 5.1e-4 | ~1.0 |
| `ModernBertModel` | `answerdotai/ModernBERT-large` | 2.4e-3 | 0.9999999 |

The kerasformers tokenizer loads ModernBERT's `tokenizer.json` directly, so it
reproduces HF's `input_ids` / `attention_mask` exactly.
