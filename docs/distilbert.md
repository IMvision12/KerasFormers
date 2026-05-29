# DistilBERT (text encoder)

Hugging Face's DistilBERT in **pure Keras 3** — the distilled (6-layer)
bidirectional transformer text encoder with its masked-LM, classification,
token-classification, QA, and multiple-choice heads. One implementation runs
unmodified on **TensorFlow / Torch / JAX**, with bit-close parity to Hugging
Face on real checkpoints (see below).

**Paper**: [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)

| Class | Module | Output |
|---|---|---|
| `DistilBertModel` | `kerasformers.models.distilbert` | `{"last_hidden_state": (B, L, embed_dim)}` |
| `DistilBertMaskedLM` | `kerasformers.models.distilbert` | MLM logits `(B, L, vocab_size)` |
| `DistilBertSequenceClassify` | `kerasformers.models.distilbert` | sequence logits `(B, num_classes)` |
| `DistilBertTokenClassify` | `kerasformers.models.distilbert` | per-token logits `(B, L, num_classes)` |
| `DistilBertQnA` | `kerasformers.models.distilbert` | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `DistilBertMultipleChoice` | `kerasformers.models.distilbert` | per-choice logits `(B, num_choices)` |
| `DistilBertTokenizer` | `kerasformers.models.distilbert` | WordPiece → `input_ids` / `attention_mask` |

All models are functional `BaseModel`s; the head classes compose a
`DistilBertModel` backbone. DistilBERT has **no pooler and no token-type
embeddings** (so the tokenizer emits only `input_ids` / `attention_mask`), and
no next-sentence head. The MLM head is part of the pretrained checkpoint and
loads real weights; the classification heads (a `pre_classifier` dense + ReLU
before the classifier) are randomly initialized for the official release
(ready for fine-tuning) and load trained weights from a `hf:` fine-tune. The
architecture is identical across variants — only the vocabulary and casing
differ.

## Loading

Two paths, both via `from_weights`:

- **Official release variant** — `from_weights("distilbert_base_uncased")`
  downloads the kerasformers-release `.weights.h5`.
- **`hf:` community fine-tune** — `from_weights("hf:org/repo")` reads the repo's
  `config.json` (architecture + `num_labels`) and loads the checkpoint, including
  the fine-tuned classifier head.

```python
from kerasformers.models.distilbert import DistilBertModel, DistilBertTokenizer

model = DistilBertModel.from_weights("distilbert_base_uncased")
tokenizer = DistilBertTokenizer.from_weights("distilbert_base_uncased")

out = model(tokenizer("Hello, world."))
out["last_hidden_state"]   # (1, L, 768)
```

### Available variants

| Variant | vocab | casing |
|---|---|---|
| `distilbert_base_uncased` | 30522 | lowercased |
| `distilbert_base_cased` | 28996 | case-preserving |
| `distilbert_base_multilingual_cased` | 119547 | case-preserving |

The tokenizer follows the model: `DistilBertTokenizer.from_weights(variant)` pulls
the matching vocab and casing, and `from_weights("hf:org/repo")` reads
`do_lower_case` from the repo's `tokenizer_config.json`.

## Verified parity

Validated against the Hugging Face reference (eager attention) on a real forward
pass:

| Model | Checkpoint | max \|Δ\| |
|---|---|---|
| `DistilBertModel` | `distilbert-base-uncased` | 3.6e-6 |
| `DistilBertMaskedLM` | `distilbert-base-uncased` | 3.6e-5 |
| `DistilBertModel` | `distilbert-base-cased` | 1.9e-6 |

The kerasformers WordPiece tokenizer reproduces HF's `input_ids` /
`attention_mask` exactly.
