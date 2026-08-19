# ModernBERT (text encoder)

<div class="kf-note kf-note--weights">
<b>Weights:</b> load a converted repo with
<code>from_weights("kerasformers/modernbert_base")</code> (each repo carries
<code>kf_config.json</code> + <code>model.weights.h5</code> + <code>tokenizer.json</code>),
or convert an upstream checkpoint on the fly with
<code>from_weights("hf:answerdotai/ModernBERT-base")</code>.
</div>

Answer.AI / LightOn's **ModernBERT** in **pure Keras 3**: a modernized
bidirectional transformer text encoder (rotary embeddings, alternating
local/global attention, GeGLU feed-forwards, pre-LayerNorm) with its masked-LM,
classification, token-classification, QA, and multiple-choice heads. One
implementation runs unmodified on **TensorFlow / Torch / JAX**, with bit-close
parity to Hugging Face on real checkpoints (see below).

**Paper**: [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663)

| Task | Class | HF equivalent | Output |
|---|---|---|---|
| Backbone | `ModernBertModel` | `ModernBertModel` | `{"last_hidden_state": (B, L, embed_dim)}` |
| Masked LM | `ModernBertMaskedLM` | `ModernBertForMaskedLM` | MLM logits `(B, L, vocab_size)` |
| Sequence classify | `ModernBertSequenceClassify` | `ModernBertForSequenceClassification` | `(B, num_classes)` |
| Token classify | `ModernBertTokenClassify` | `ModernBertForTokenClassification` | `(B, L, num_classes)` |
| Question answering | `ModernBertQnA` | `ModernBertForQuestionAnswering` | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| Multiple choice | `ModernBertMultipleChoice` | `ModernBertForMultipleChoice` | `(B, num_choices)` |
| Tokenizer | `ModernBertTokenizer` | `ModernBertTokenizerFast` | `input_ids` / `attention_mask` |

All classes live in `kerasformers.models.modernbert` and are functional
`FunctionalBaseModel`s; the head classes compose a `ModernBertModel` backbone.
ModernBERT has **no pooler and no token-type embeddings** (position is injected by
rotary embeddings), so the tokenizer emits only `input_ids` / `attention_mask`.

The one hosted `model.weights.h5` (declaring `ModernBertModel`) is the full
masked-LM checkpoint (encoder + prediction head + tied decoder); every class loads
its own subset via `CHECKPOINT_SOURCE`. The shared prediction head
(`head_dense` / `head_norm`) is pretrained, so it also feeds the classification
heads (matching Hugging Face); their final classifier / span layer is randomly
initialized, ready for fine-tuning, and loads trained weights from a `hf:`
fine-tune. The architecture is identical across variants, only the depth and width
differ.

## Loading

```python
from kerasformers.models.modernbert import ModernBertModel, ModernBertTokenizer

model = ModernBertModel.from_weights("kerasformers/modernbert_base")
tokenizer = ModernBertTokenizer.from_weights("kerasformers/modernbert_base")

out = model(tokenizer("Hello, world."))
out["last_hidden_state"]  # (1, L, 768)
```

Or convert on the fly from the upstream checkpoint with
`from_weights("hf:answerdotai/ModernBERT-base")`. The `hf:` path reads the repo's
`config.json` (architecture + `num_labels`) and loads the checkpoint, including a
fine-tuned classifier head for a community fine-tune.

### Available variants

| Variant | layers | embed_dim | heads | mlp_dim |
|---|---|---|---|---|
| `modernbert_base` | 22 | 768 | 12 | 1152 |
| `modernbert_large` | 28 | 1024 | 16 | 2624 |

Both variants share one tokenizer
(`ModernBertTokenizer.from_weights("kerasformers/<variant>")`).

## Fill-mask

```python
from kerasformers.models.modernbert import ModernBertMaskedLM, ModernBertTokenizer

mlm = ModernBertMaskedLM.from_weights("kerasformers/modernbert_base")
tokenizer = ModernBertTokenizer.from_weights("kerasformers/modernbert_base")
logits = mlm(tokenizer("The capital of France is [MASK]."))  # (1, L, vocab_size)
```

## Architecture notes

- **Rotary position embeddings** with two bases: global layers use
  `global_rope_theta=160000`, local layers use `local_rope_theta=10000`.
- **Alternating attention**: every `global_attn_every_n_layers` (3rd) layer uses
  full attention; the rest use a sliding window of `local_attention` (128) tokens.
- **GeGLU** feed-forward (`Wi` projects to `2 * mlp_dim`, gated, then `Wo`).
- **Pre-LayerNorm** residuals; bias-free linears and LayerNorms; the first
  layer's attention LayerNorm is the identity (the embeddings are already
  normalized). Each stack ends with a final LayerNorm.
- The MLM decoder is **tied** to the token embeddings.

## Parity

Validated against the Hugging Face reference (eager attention) on a real forward
pass with a sequence long enough to exercise the sliding-window (local)
attention. The larger max residual on `large` is fp32 op-order accumulation over
the deeper/wider stack (mean residual is ~6e-6 and cosine is ~1.0), not an
architectural difference, so the converter gates on cosine (>= 0.9999), like the
deep DeBERTa-v2 models.

| Model | Checkpoint | max \|Δ\| | cosine |
|---|---|---|---|
| `ModernBertModel` | `answerdotai/ModernBERT-base` | 2.0e-4 | ~1.0 |
| `ModernBertMaskedLM` | `answerdotai/ModernBERT-base` | 5.1e-4 | ~1.0 |
| `ModernBertModel` | `answerdotai/ModernBERT-large` | 2.4e-3 | 0.9999999 |

The kerasformers tokenizer loads ModernBERT's `tokenizer.json` directly, so it
reproduces HF's `input_ids` / `attention_mask` exactly.
