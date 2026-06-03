# RoBERTa (text encoder)

Facebook AI's RoBERTa in **pure Keras 3** — the robustly-optimized bidirectional
transformer text encoder with its masked-LM, sequence-classification,
token-classification, question-answering, and multiple-choice heads. One
implementation runs unmodified on **TensorFlow / Torch / JAX**, with bit-close
parity to Hugging Face on real checkpoints (see below).

**Paper**: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

| Class | Module | Output |
|---|---|---|
| `RobertaModel` | `kerasformers.models.roberta` | `{"last_hidden_state": (B, L, embed_dim), "pooler_output": (B, embed_dim)}` |
| `RobertaMaskedLM` | `kerasformers.models.roberta` | MLM logits `(B, L, vocab_size)` |
| `RobertaSequenceClassify` | `kerasformers.models.roberta` | sequence logits `(B, num_classes)` |
| `RobertaTokenClassify` | `kerasformers.models.roberta` | per-token logits `(B, L, num_classes)` |
| `RobertaQnA` | `kerasformers.models.roberta` | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `RobertaMultipleChoice` | `kerasformers.models.roberta` | per-choice logits `(B, num_choices)` |
| `RobertaTokenizer` | `kerasformers.models.roberta` | byte-level BPE → `input_ids` / `attention_mask` / `token_type_ids` |

All models are functional `FunctionalBaseModel`s; the head classes compose a `RobertaModel`
backbone. The masked-LM head is part of the pretrained checkpoint, so it loads
real weights; the other task heads are randomly initialized for the official
release (ready for fine-tuning) and load trained weights from a `hf:` fine-tune.

RoBERTa shares BERT's encoder but differs in three ways: position ids are offset
by the padding id (so the table has two extra slots), there is a single
token-type, and the sentence classifier reads the `<s>` token directly through a
`tanh` head rather than a separate pooler. There is **no next-sentence head**.

## Loading

Two paths, both via `from_weights`:

- **Official release variant** — `from_weights("roberta_base")` downloads the
  kerasformers-release `.weights.h5`.
- **`hf:` community fine-tune** — `from_weights("hf:org/repo")` reads the repo's
  `config.json` (architecture + `num_labels`) and loads the checkpoint, including
  the fine-tuned classifier head.

```python
from kerasformers.models.roberta import RobertaModel, RobertaTokenizer

model = RobertaModel.from_weights("roberta_base")
tokenizer = RobertaTokenizer.from_weights("roberta_base")

out = model(tokenizer("Hello, world."))
out["last_hidden_state"]   # (1, L, 768)
out["pooler_output"]       # (1, 768)
```

### Available variants

| Variant | vocab | layers / dim |
|---|---|---|
| `roberta_base` | 50265 | 12 / 768 |
| `roberta_large` | 50265 | 24 / 1024 |

## Verified parity

Validated against the Hugging Face reference (eager attention) on a real forward
pass, including a padded sequence so the padding-offset position ids are
exercised:

| Model | Checkpoint | max \|Δ\| |
|---|---|---|
| `RobertaModel` | `FacebookAI/roberta-base` | 7.6e-6 |
| `RobertaMaskedLM` | `FacebookAI/roberta-base` | 3.2e-5 |
| `RobertaSequenceClassify` | `textattack/roberta-base-SST-2` (`hf:`) | 6.7e-7 |
| `RobertaQnA` | `deepset/roberta-base-squad2` (`hf:`) | 1.1e-5 |

The kerasformers byte-level BPE tokenizer reproduces HF's `input_ids` /
`token_type_ids` / `attention_mask` exactly (single, pair, and whitespace-heavy
inputs).

## Forward pass

The models take a dict of token ids, an attention mask, and segment ids — exactly
what `RobertaTokenizer` returns (segment ids are always `0` for RoBERTa):

```python
inputs = {
    "input_ids":      input_ids,       # (B, L) int
    "attention_mask": attention_mask,  # (B, L) int — 1 keep, 0 pad
    "token_type_ids": token_type_ids,  # (B, L) int — all 0
}
RobertaModel.from_weights("roberta_base")(inputs)["last_hidden_state"]
```

These are token-id models — **no spatial H/W axes**, so `channels_first/last`
does not apply.

### Fill-mask

```python
from kerasformers.models.roberta import RobertaMaskedLM, RobertaTokenizer

mlm = RobertaMaskedLM.from_weights("roberta_base")
tokenizer = RobertaTokenizer.from_weights("roberta_base")

inputs = tokenizer("The capital of France is <mask>.")
logits = mlm(inputs)                                  # (1, L, vocab_size)
mask = int((inputs["input_ids"][0] == tokenizer.mask_token_id).argmax())
print(tokenizer.decode([int(logits[0, mask].argmax())]))   # -> " Paris"
```

### Classification (community fine-tunes)

```python
from kerasformers.models.roberta import RobertaSequenceClassify, RobertaTokenClassify

# sentiment (2 labels)
clf = RobertaSequenceClassify.from_weights("hf:textattack/roberta-base-SST-2")
# named-entity recognition
ner = RobertaTokenClassify.from_weights("hf:Jean-Baptiste/roberta-large-ner-english")
```

`num_classes` is read from the repo's config, so the head matches the fine-tune.
For example, `RobertaSequenceClassify.from_weights("hf:FacebookAI/roberta-large-mnli")`
loads the 3-class NLI model (`0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT`), commonly
used for zero-shot classification.

### Other task heads

```python
from kerasformers.models.roberta import RobertaQnA, RobertaMultipleChoice

# extractive QA — start/end span logits, from a SQuAD fine-tune
qa = RobertaQnA.from_weights("hf:deepset/roberta-base-squad2")
out = qa(tokenizer("Where is Paris?", text_pair="Paris is in France."))
out["start_logits"]  # (B, L)   out["end_logits"]  # (B, L)

# multiple choice — inputs are (B, num_choices, seq); fix num_choices at build
mc = RobertaMultipleChoice.from_weights("hf:org/roberta-swag", num_choices=4)
```

`RobertaMultipleChoice` takes a static `num_choices` (the choice axis is folded
into the batch through the shared backbone, then back out); its `classifier` head
is shape-independent of `num_choices`, so the same weights load for any value.

## Tokenizer

`RobertaTokenizer` is a byte-level BPE tokenizer built on the `tokenizers` (Rust)
library: no normalization, byte-level pre-tokenization, BPE over `vocab.json` +
`merges.txt`, and `<s> A </s>` / `<s> A </s> </s> B </s>` RoBERTa-style
post-processing, so encoding matches `AutoTokenizer` exactly (the left-stripping
`<mask>` token is protected, so fill-mask inputs work). It accepts a string, a
list of strings, or a sentence pair (`text_pair=`), and returns the `input_ids` /
`attention_mask` / `token_type_ids` dict the models consume.

## Architecture notes

- **Embeddings** — summed word + absolute-position + token-type embeddings, then
  LayerNorm + dropout. Position ids are derived from the non-padding mask
  (`cumsum(input_ids != pad) * mask + pad`) so padding tokens map to the padding
  slot and the first real token starts at `pad + 1`; this is computed with masked
  `cumsum` rather than `arange`, keeping the model shape-polymorphic across
  backends.
- **Encoder** — `num_layers` post-LayerNorm transformer blocks
  (`LayerNorm(x + Sublayer(x))`): multi-head self-attention with an additive
  padding mask, then a `mlp_dim` feed-forward with `hidden_act` (exact `gelu`).
  `layer_norm_eps` is `1e-5`.
- **Pooler** — a `tanh` dense projection of the `<s>` token (`RobertaModel`).
- **Heads** — `RobertaMaskedLM` adds the transform (dense + `gelu` + LayerNorm)
  and a vocabulary projection; `RobertaSequenceClassify` uses RoBERTa's
  classification head (dropout + `tanh` dense + dropout + projection) on the
  `<s>` token; the other classify models add dropout + a dense classifier.

Constructor arguments follow the kerasformers convention: `embed_dim`,
`num_layers`, `num_heads`, `mlp_dim` (plus `vocab_size`,
`max_position_embeddings`, `type_vocab_size`, `hidden_act`, `layer_norm_eps`,
`pad_token_id`). `convert_roberta_hf_to_keras.py` maps the HF `RobertaModel` /
`RobertaForMaskedLM` (and the task heads) safetensors to Keras.

## Citation

```bibtex
@article{RoBERTa, title={RoBERTa: A Robustly Optimized BERT Pretraining Approach}, author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and Zettlemoyer, Luke and Stoyanov, Veselin}, journal={arXiv:1907.11692}, year={2019}}
```
