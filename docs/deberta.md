# DeBERTa v1 / v2 / v3 (text encoder)

Microsoft's DeBERTa family in **pure Keras 3** — the disentangled-attention
bidirectional text encoder, in its three generations (v1, v2, v3), each with the
masked-LM, sequence-classification, token-classification, question-answering, and
multiple-choice heads. One implementation per version runs unmodified on
**TensorFlow / Torch / JAX**, with bit-close parity to Hugging Face on real
checkpoints (see below).

**Papers**: [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
(v1 / v2) · [DeBERTaV3: ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543)
(v3)

Each generation lives in its own package with the same set of model classes plus a
tokenizer; only the module path and the class prefix change (v1 has no
multiple-choice head):

| Task | v1 — `kerasformers.models.deberta` | v2 — `kerasformers.models.deberta_v2` | v3 — `kerasformers.models.deberta_v3` | Output |
|---|---|---|---|---|
| Backbone | `DebertaModel` | `DebertaV2Model` | `DebertaV3Model` | `{"last_hidden_state": (B, L, embed_dim)}` |
| Masked LM | `DebertaMaskedLM` | `DebertaV2MaskedLM` | `DebertaV3MaskedLM` | MLM logits `(B, L, vocab_size)` |
| Sequence classify | `DebertaSequenceClassify` | `DebertaV2SequenceClassify` | `DebertaV3SequenceClassify` | `(B, num_classes)` |
| Token classify | `DebertaTokenClassify` | `DebertaV2TokenClassify` | `DebertaV3TokenClassify` | `(B, L, num_classes)` |
| Question answering | `DebertaQnA` | `DebertaV2QnA` | `DebertaV3QnA` | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| Multiple choice | — | `DebertaV2MultipleChoice` | `DebertaV3MultipleChoice` | `(B, num_choices)` |
| Tokenizer | `DebertaTokenizer` (byte-level BPE) | `DebertaV2Tokenizer` (SentencePiece) | `DebertaV3Tokenizer` (SentencePiece) | `input_ids` / `attention_mask` / `token_type_ids` |

All models are functional `BaseModel`s; the head classes compose the matching
backbone. Unlike BERT, **DeBERTa has no pooler and no next-sentence head** — the
backbone returns only `last_hidden_state`, and the sequence/multiple-choice heads
attach DeBERTa's *context pooler* (a dense + `gelu` over the `[CLS]` token). The
masked-LM head is part of the pretrained checkpoint; the other task heads are
randomly initialized for the official release (ready for fine-tuning) and load
trained weights from a `hf:` fine-tune.

## Loading

Two paths, both via `from_weights`:

- **Official release variant** — `from_weights("deberta_base")` downloads the
  kerasformers-release `.weights.h5`.
- **`hf:` checkpoint / community fine-tune** — `from_weights("hf:org/repo")` reads
  the repo's `config.json` (architecture + `num_labels`) and loads the checkpoint,
  including a fine-tuned classifier head.

```python
from kerasformers.models.deberta_v3 import DebertaV3Model, DebertaV3Tokenizer

model = DebertaV3Model.from_weights("deberta_v3_base")
tokenizer = DebertaV3Tokenizer.from_weights("deberta_v3_base")

out = model(tokenizer("Hello, world."))
out["last_hidden_state"]   # (1, L, 768)
```

### Available variants

| Version | Variant | vocab | layers / dim |
|---|---|---|---|
| v1 | `deberta_base` | 50265 | 12 / 768 |
| v1 | `deberta_large` | 50265 | 24 / 1024 |
| v2 | `deberta_v2_xlarge` | 128100 | 24 / 1536 |
| v2 | `deberta_v2_xxlarge` | 128100 | 48 / 1536 |
| v3 | `deberta_v3_xsmall` | 128100 | 12 / 384 |
| v3 | `deberta_v3_small` | 128100 | 6 / 768 |
| v3 | `deberta_v3_base` | 128100 | 12 / 768 |
| v3 | `deberta_v3_large` | 128100 | 24 / 1024 |

## Verified parity

Validated against the Hugging Face reference (eager attention) on a real forward
pass, including a padded sequence:

| Model | Checkpoint | max \|Δ\| |
|---|---|---|
| `DebertaModel` | `microsoft/deberta-base` | 2.6e-5 |
| `DebertaMaskedLM` | `microsoft/deberta-base` | 4.8e-5 |
| `DebertaV2Model` | `microsoft/deberta-v2-xlarge` | 4.8e-7 |
| `DebertaV2MaskedLM` | `microsoft/deberta-v2-xlarge` | 1.8e-7 |
| `DebertaV3Model` | `microsoft/deberta-v3-base` | 9.7e-5 |
| `DebertaV3Model` | `microsoft/deberta-v3-large` | 2.9e-6 |

> **DeBERTa-v3 checkpoints ship as float16.** Load the HF reference with
> `from_pretrained(..., dtype=torch.float32)` to compare like-for-like — otherwise
> HF runs the forward in fp16 and the ~0.07 gap is the fp16/fp32 difference, not a
> model error (Keras runs in fp32, so it is actually closer to the fp64 ideal).

## Forward pass

The models take a dict of token ids, an attention mask, and segment ids — exactly
what the tokenizer returns (segment ids are always `0`; DeBERTa has no token-type
embeddings and ignores them, they are accepted for API parity):

```python
inputs = {
    "input_ids":      input_ids,       # (B, L) int
    "attention_mask": attention_mask,  # (B, L) int — 1 keep, 0 pad
    "token_type_ids": token_type_ids,  # (B, L) int — all 0 (unused)
}
DebertaV3Model.from_weights("deberta_v3_base")(inputs)["last_hidden_state"]
```

These are token-id models — **no spatial H/W axes**, so `channels_first/last`
does not apply.

### Fill-mask

```python
from kerasformers.models.deberta_v3 import DebertaV3MaskedLM, DebertaV3Tokenizer

mlm = DebertaV3MaskedLM.from_weights("deberta_v3_base")
tokenizer = DebertaV3Tokenizer.from_weights("deberta_v3_base")

inputs = tokenizer("The capital of France is [MASK].")
logits = mlm(inputs)                                  # (1, L, vocab_size)
mask = int((inputs["input_ids"][0] == tokenizer.mask_token_id).argmax())
print(tokenizer.decode([int(logits[0, mask].argmax())]))
```

### Classification (community fine-tunes)

```python
from kerasformers.models.deberta_v3 import (
    DebertaV3SequenceClassify,
    DebertaV3TokenClassify,
)

# natural-language inference / zero-shot (3 labels)
nli = DebertaV3SequenceClassify.from_weights("hf:microsoft/deberta-v3-base")  # + a fine-tune
# named-entity recognition
ner = DebertaV3TokenClassify.from_weights("hf:org/deberta-v3-ner")
```

`num_classes` is read from the repo's config, so the head matches the fine-tune.

### Other task heads

```python
from kerasformers.models.deberta_v3 import DebertaV3QnA, DebertaV3MultipleChoice

# extractive QA — start/end span logits
qa = DebertaV3QnA.from_weights("hf:org/deberta-v3-squad")
out = qa(tokenizer("Where is Paris?", text_pair="Paris is in France."))
out["start_logits"]  # (B, L)   out["end_logits"]  # (B, L)

# multiple choice — inputs are (B, num_choices, seq); fix num_choices at build
mc = DebertaV3MultipleChoice.from_weights("hf:org/deberta-v3-swag", num_choices=4)
```

`*MultipleChoice` takes a static `num_choices` (the choice axis is folded into the
batch through the shared backbone, then back out); its `classifier` head is
shape-independent of `num_choices`, so the same weights load for any value.

## Tokenizers

- **v1 — `DebertaTokenizer`** is a byte-level BPE tokenizer (`vocab.json` +
  `merges.txt`, like GPT-2 / RoBERTa) but with BERT-style specials
  (`[CLS] A [SEP]`, and `[CLS] A [SEP] [SEP] B [SEP]` for pairs).
- **v2 / v3 — `DebertaV2Tokenizer` / `DebertaV3Tokenizer`** are SentencePiece
  Unigram tokenizers (`spm.model`, 128 100 pieces, no fairseq id offset) with
  `[CLS] A [SEP]` / `[CLS] A [SEP] B [SEP]` post-processing. v3 differs from v2
  only in the underlying `spm.model`.

All return the `input_ids` / `attention_mask` / `token_type_ids` dict the models
consume, and reproduce Hugging Face's ids exactly (single, pair, and
whitespace-heavy inputs).

## Architecture notes

DeBERTa replaces BERT's absolute-position embeddings with **disentangled
attention**: each token is represented by separate content and (relative)
position vectors, and the attention score sums content→content,
content→position (c2p), and position→content (p2c) terms, scaled by
`1/sqrt(head_dim · (1 + #pos_att_type))`. Position information enters only through
attention — the input embeddings are word embeddings + LayerNorm only (no
absolute-position or token-type embeddings).

- **v1** — fused `in_proj` query/key/value with separate query/value biases, and
  dedicated `pos_proj` (c2p) / `pos_q_proj` (p2c) projections over a raw
  relative-position embedding table (`rel[i,j] = i − j`). `layer_norm_eps = 1e-7`.
- **v2** — separate `query_proj` / `key_proj` / `value_proj`; **log-bucketed**
  relative positions (`position_buckets = 256`) so a small table spans long
  sequences; `share_att_key` reuses the content projections for the relative
  terms; a LayerNorm on the relative embeddings (`norm_rel_ebd`); and a single
  depthwise-style **convolution** (`conv_kernel_size = 3`) after the first encoder
  layer. SentencePiece vocabulary (128 100).
- **v3** — the v2 architecture (HF `model_type = "deberta-v2"`, so it reuses the
  v2 backbone) **without the convolution**, pretrained ELECTRA-style with
  gradient-disentangled embedding sharing. Only the backbone is ported; the
  replaced-token-detection and mask-prediction heads are discriminator-only and
  not used for the encoder.

Constructor arguments follow the kerasformers convention: `embed_dim`,
`num_layers`, `num_heads`, `mlp_dim` (plus `vocab_size`,
`max_position_embeddings`, `max_relative_positions`, `pos_att_type`,
`hidden_act`, `layer_norm_eps`, `pad_token_id`; v2 / v3 add `position_buckets`,
`norm_rel_ebd`, `conv_kernel_size`, `conv_act`). Each
`convert_deberta*_hf_to_keras.py` maps the HF checkpoint's weights to Keras.

## Citation

```bibtex
@inproceedings{he2021deberta, title={DeBERTa: Decoding-enhanced BERT with Disentangled Attention}, author={He, Pengcheng and Liu, Xiaodong and Gao, Jianfeng and Chen, Weizhu}, booktitle={ICLR}, year={2021}}
@inproceedings{he2023debertav3, title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing}, author={He, Pengcheng and Gao, Jianfeng and Chen, Weizhu}, booktitle={ICLR}, year={2023}}
```
