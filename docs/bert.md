# BERT (text encoder)

Google's BERT in **pure Keras 3**: the bidirectional transformer text encoder
with its masked-LM, sequence-classification, and token-classification heads.
One implementation runs unmodified on **TensorFlow / Torch / JAX**, with
bit-close parity to Hugging Face on real checkpoints (see below).

**Paper**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

| Class | Module | Output |
|---|---|---|
| `BertModel` | `kerasformers.models.bert` | `{"last_hidden_state": (B, L, embed_dim), "pooler_output": (B, embed_dim)}` |
| `BertMaskedLM` | `kerasformers.models.bert` | MLM logits `(B, L, vocab_size)` |
| `BertSequenceClassify` | `kerasformers.models.bert` | sequence logits `(B, num_classes)` |
| `BertTokenClassify` | `kerasformers.models.bert` | per-token logits `(B, L, num_classes)` |
| `BertNextSentencePredict` | `kerasformers.models.bert` | next-sentence logits `(B, 2)` |
| `BertQnA` | `kerasformers.models.bert` | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `BertMultipleChoice` | `kerasformers.models.bert` | per-choice logits `(B, num_choices)` |
| `BertTokenizer` | `kerasformers.models.bert` | WordPiece → `input_ids` / `attention_mask` / `token_type_ids` |

All models are functional `FunctionalBaseModel`s; the head classes compose a `BertModel`
backbone. The next-sentence head is part of the pretrained checkpoint, so it
loads real weights; the other task heads are randomly initialized for the
official release (ready for fine-tuning) and load trained weights from a `hf:`
fine-tune. The architecture is identical across variants: only the vocabulary
and tokenizer casing differ.

## Loading

Two paths, both via `from_weights`:

- **Official release variant**: `from_weights("bert_base_uncased")` downloads the
  kerasformers-release `.weights.h5`.
- **`hf:` community fine-tune**: `from_weights("hf:org/repo")` reads the repo's
  `config.json` (architecture + `num_labels`) and loads the checkpoint, including
  the fine-tuned classifier head.

```python
from kerasformers.models.bert import BertModel, BertTokenizer

model = BertModel.from_weights("bert_base_uncased")
tokenizer = BertTokenizer.from_weights("bert_base_uncased")

out = model(tokenizer("Hello, world."))
out["last_hidden_state"]   # (1, L, 768)
out["pooler_output"]       # (1, 768)
```

### Available variants

| Variant | vocab | casing |
|---|---|---|
| `bert_base_uncased` | 30522 | lowercased |
| `bert_large_uncased` | 30522 | lowercased |
| `bert_base_cased` | 28996 | case-preserving |
| `bert_large_cased` | 28996 | case-preserving |

The tokenizer follows the model: `BertTokenizer.from_weights(variant)` pulls the
matching vocab and casing, and `from_weights("hf:org/repo")` reads `do_lower_case`
from the repo's `tokenizer_config.json`.

## Verified parity

Validated against the Hugging Face reference (eager attention) on a real forward
pass (the classification fine-tunes also agree with HF on `argmax` at every
position):

| Model | Checkpoint | max \|Δ\| |
|---|---|---|
| `BertModel` | `google-bert/bert-base-uncased` | 1.2e-5 |
| `BertModel` | `google-bert/bert-base-cased` | 6.1e-6 |
| `BertMaskedLM` | `google-bert/bert-base-uncased` | 1.7e-5 |
| `BertSequenceClassify` | `textattack/bert-base-uncased-SST-2` (`hf:`) | 4.8e-7 |
| `BertTokenClassify` | `dslim/bert-base-NER` (`hf:`) | 2.9e-6 |
| `BertNextSentencePredict` | `google-bert/bert-base-uncased` | 4.8e-6 |
| `BertQnA` | `deepset/bert-base-cased-squad2` (`hf:`) | 4.3e-6 |
| `BertMultipleChoice` | `google-bert/bert-base-uncased` | 3.3e-7 |

The kerasformers WordPiece tokenizer reproduces HF's `input_ids` /
`token_type_ids` / `attention_mask` exactly (single, pair, and `[MASK]` inputs;
both casings).

## Forward pass

The models take a dict of token ids, an attention mask, and segment ids: exactly
what `BertTokenizer` returns:

```python
inputs = {
    "input_ids":      input_ids,       # (B, L) int
    "attention_mask": attention_mask,  # (B, L) int: 1 keep, 0 pad
    "token_type_ids": token_type_ids,  # (B, L) int: segment (0 / 1)
}
BertModel.from_weights("bert_base_uncased")(inputs)["last_hidden_state"]
```

These are token-id models: **no spatial H/W axes**, so `channels_first/last`
does not apply.

### Fill-mask

```python
from kerasformers.models.bert import BertMaskedLM, BertTokenizer

mlm = BertMaskedLM.from_weights("bert_base_uncased")
tokenizer = BertTokenizer.from_weights("bert_base_uncased")

inputs = tokenizer("the capital of france is [MASK].")
logits = mlm(inputs)                                  # (1, L, vocab_size)
mask = int((inputs["input_ids"][0] == tokenizer.mask_token_id).argmax())
print(tokenizer.ids_to_tokens[int(logits[0, mask].argmax())])   # -> "paris"
```

### Classification (community fine-tunes)

```python
from kerasformers.models.bert import BertSequenceClassify, BertTokenClassify

# sentiment (2 labels)
clf = BertSequenceClassify.from_weights("hf:textattack/bert-base-uncased-SST-2")
# named-entity recognition (9 labels, cased)
ner = BertTokenClassify.from_weights("hf:dslim/bert-base-NER")
```

`num_classes` is read from the repo's config, so the head matches the fine-tune.

### Other task heads

```python
from kerasformers.models.bert import (
    BertNextSentencePredict, BertQnA, BertMultipleChoice,
)

# next-sentence prediction: head is pretrained, so a base checkpoint works
nsp = BertNextSentencePredict.from_weights("hf:google-bert/bert-base-uncased")

# extractive QA: start/end span logits, from a SQuAD fine-tune
qa = BertQnA.from_weights("hf:deepset/bert-base-cased-squad2")
out = qa(tokenizer("Where is Paris?", text_pair="Paris is in France."))
out["start_logits"]  # (B, L)   out["end_logits"]  # (B, L)

# multiple choice: inputs are (B, num_choices, seq); fix num_choices at build
mc = BertMultipleChoice.from_weights("hf:org/bert-swag", num_choices=4)
```

`BertMultipleChoice` takes a static `num_choices` (the choice axis is folded into
the batch through the shared backbone, then back out); its `classifier` head is
shape-independent of `num_choices`, so the same weights load for any value.

## Tokenizer

`BertTokenizer` is a WordPiece tokenizer built on the `tokenizers` (Rust) library:
the BERT normalizer (clean text, handle CJK, optional lowercase + accent strip),
whitespace/punctuation pre-tokenization, greedy WordPiece over `vocab.txt`, and
`[CLS] A [SEP] B [SEP]` template post-processing with segment ids. It accepts a
string, a list of strings, or a sentence pair (`text_pair=`), and returns the
`input_ids` / `attention_mask` / `token_type_ids` dict the models consume.

## Architecture notes

- **Embeddings**: summed word + absolute-position + token-type embeddings, then
  LayerNorm + dropout. Position ids come from `cumsum(ones_like(input_ids)) - 1`
  (rather than `arange`) so the model stays shape-polymorphic across backends and
  supports variable sequence length.
- **Encoder**: `num_layers` post-LayerNorm transformer blocks
  (`LayerNorm(x + Sublayer(x))`): multi-head self-attention with an additive
  padding mask, then a `mlp_dim` feed-forward with `hidden_act` (exact `gelu`).
- **Pooler**: a `tanh` dense projection of the `[CLS]` token (`BertModel`,
  `BertSequenceClassify`).
- **Heads**: `BertMaskedLM` adds the transform (dense + `gelu` + LayerNorm) and a
  vocabulary projection; the classify models add dropout + a dense classifier.

Constructor arguments follow the kerasformers convention: `embed_dim`,
`num_layers`, `num_heads`, `mlp_dim` (plus `vocab_size`,
`max_position_embeddings`, `type_vocab_size`, `hidden_act`, `layer_norm_eps`).
`convert_bert_hf_to_keras.py` maps the HF `BertModel` / `BertForMaskedLM`
safetensors to Keras.

## Citation

```bibtex
@article{BERT, title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding}, author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina}, journal={arXiv:1810.04805}, year={2018}}
```
