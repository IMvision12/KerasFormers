# XLM-RoBERTa (multilingual text encoder)

Facebook AI's XLM-RoBERTa in **pure Keras 3** — the multilingual variant of
RoBERTa, pretrained on 2.5TB of filtered CommonCrawl across 100 languages, with
its masked-LM, sequence-classification, token-classification,
question-answering, and multiple-choice heads. One implementation runs
unmodified on **TensorFlow / Torch / JAX**, with bit-close parity to Hugging Face
on real checkpoints (see below).

**Paper**: [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)

| Class | Module | Output |
|---|---|---|
| `XLMRobertaModel` | `kerasformers.models.xlm_roberta` | `{"last_hidden_state": (B, L, embed_dim), "pooler_output": (B, embed_dim)}` |
| `XLMRobertaMaskedLM` | `kerasformers.models.xlm_roberta` | MLM logits `(B, L, vocab_size)` |
| `XLMRobertaSequenceClassify` | `kerasformers.models.xlm_roberta` | sequence logits `(B, num_classes)` |
| `XLMRobertaTokenClassify` | `kerasformers.models.xlm_roberta` | per-token logits `(B, L, num_classes)` |
| `XLMRobertaQnA` | `kerasformers.models.xlm_roberta` | `{"start_logits": (B, L), "end_logits": (B, L)}` |
| `XLMRobertaMultipleChoice` | `kerasformers.models.xlm_roberta` | per-choice logits `(B, num_choices)` |
| `XLMRobertaTokenizer` | `kerasformers.models.xlm_roberta` | SentencePiece → `input_ids` / `attention_mask` / `token_type_ids` |

XLM-RoBERTa is **architecturally identical to RoBERTa** — it reuses the same
encoder, padding-offset position ids, single token-type, `1e-5` LayerNorm, and
head structures — and differs only in scale: a 250k multilingual SentencePiece
vocabulary instead of RoBERTa's 50k byte-level BPE. All models are functional
`BaseModel`s; the head classes compose an `XLMRobertaModel` backbone.

## Loading

Two paths, both via `from_weights`:

- **Official release variant** — `from_weights("xlm_roberta_base")` downloads the
  kerasformers-release `.weights.h5`.
- **`hf:` community fine-tune** — `from_weights("hf:org/repo")` reads the repo's
  `config.json` (architecture + `num_labels`) and loads the checkpoint, including
  the fine-tuned classifier head.

```python
from kerasformers.models.xlm_roberta import XLMRobertaModel, XLMRobertaTokenizer

model = XLMRobertaModel.from_weights("xlm_roberta_base")
tokenizer = XLMRobertaTokenizer.from_weights("xlm_roberta_base")

out = model(tokenizer(["Hello, world.", "Bonjour le monde."]))
out["last_hidden_state"]   # (2, L, 768)
out["pooler_output"]       # (2, 768)
```

### Available variants

| Variant | vocab | layers / dim |
|---|---|---|
| `xlm_roberta_base` | 250002 | 12 / 768 |
| `xlm_roberta_large` | 250002 | 24 / 1024 |

## Verified parity

Validated against the Hugging Face reference (eager attention) on a real forward
pass, including a padded sequence so the padding-offset position ids are
exercised:

| Model | Checkpoint | max \|Δ\| |
|---|---|---|
| `XLMRobertaModel` | `FacebookAI/xlm-roberta-base` | 5.7e-6 |
| `XLMRobertaMaskedLM` | `FacebookAI/xlm-roberta-base` | 6.1e-5 |

The kerasformers SentencePiece tokenizer reproduces HF's `input_ids` /
`token_type_ids` / `attention_mask` exactly across languages (Latin, German,
French, and Japanese inputs were checked).

## Forward pass

The models take a dict of token ids, an attention mask, and segment ids — exactly
what `XLMRobertaTokenizer` returns (segment ids are always `0`):

```python
inputs = {
    "input_ids":      input_ids,       # (B, L) int
    "attention_mask": attention_mask,  # (B, L) int — 1 keep, 0 pad
    "token_type_ids": token_type_ids,  # (B, L) int — all 0
}
XLMRobertaModel.from_weights("xlm_roberta_base")(inputs)["last_hidden_state"]
```

These are token-id models — **no spatial H/W axes**, so `channels_first/last`
does not apply.

### Fill-mask (multilingual)

```python
from kerasformers.models.xlm_roberta import XLMRobertaMaskedLM, XLMRobertaTokenizer

mlm = XLMRobertaMaskedLM.from_weights("xlm_roberta_base")
tokenizer = XLMRobertaTokenizer.from_weights("xlm_roberta_base")

inputs = tokenizer("La capitale de la France est <mask>.")
logits = mlm(inputs)                                  # (1, L, vocab_size)
mask = int((inputs["input_ids"][0] == tokenizer.mask_token_id).argmax())
print(tokenizer.decode([int(logits[0, mask].argmax())]))   # -> "Paris"
```

### Classification (community fine-tunes)

```python
from kerasformers.models.xlm_roberta import (
    XLMRobertaSequenceClassify, XLMRobertaTokenClassify,
)

# multilingual sentiment
clf = XLMRobertaSequenceClassify.from_weights(
    "hf:cardiffnlp/twitter-xlm-roberta-base-sentiment"
)
# multilingual named-entity recognition
ner = XLMRobertaTokenClassify.from_weights("hf:Davlan/xlm-roberta-base-ner-hrl")
```

`num_classes` is read from the repo's config, so the head matches the fine-tune.

### Other task heads

```python
from kerasformers.models.xlm_roberta import XLMRobertaQnA, XLMRobertaMultipleChoice

# extractive QA — start/end span logits, from a multilingual SQuAD fine-tune
qa = XLMRobertaQnA.from_weights("hf:deepset/xlm-roberta-base-squad2")

# multiple choice — inputs are (B, num_choices, seq); fix num_choices at build
mc = XLMRobertaMultipleChoice.from_weights("hf:org/xlm-roberta-xcopa", num_choices=2)
```

`XLMRobertaMultipleChoice` takes a static `num_choices` (the choice axis is folded
into the batch through the shared backbone, then back out); its `classifier` head
is shape-independent of `num_choices`, so the same weights load for any value.

## Tokenizer

`XLMRobertaTokenizer` is a SentencePiece tokenizer built on the `tokenizers`
(Rust) library. It reads XLM-RoBERTa's `sentencepiece.bpe.model`, remaps the
pieces with the fairseq id offset (`<s>`=0, `<pad>`=1, `</s>`=2, `<unk>`=3, every
SentencePiece id shifted by +1, `<mask>` last), and reuses the model's own
`Precompiled` normalizer + `▁` metaspace pre-tokenizer with `<s> A </s>` /
`<s> A </s> </s> B </s>` post-processing, so encoding matches `AutoTokenizer`
exactly. It accepts a string, a list of strings, or a sentence pair (`text_pair=`),
and returns the `input_ids` / `attention_mask` / `token_type_ids` dict the models
consume.

## Architecture notes

XLM-RoBERTa reuses RoBERTa's encoder verbatim (`roberta_backbone`) — see the
[RoBERTa notes](roberta.md) for the embeddings (masked-`cumsum` position offset),
post-LayerNorm encoder blocks (`1e-5` epsilon, exact `gelu`), `<s>` pooler, and
head structures. The only differences are scale (`vocab_size=250002`,
`embed_dim` 768 / 1024) and the SentencePiece tokenizer.
`convert_xlm_roberta_hf_to_keras.py` reuses the RoBERTa weight transfer (the HF
backbone is exposed as `roberta.*` in both).

## Citation

```bibtex
@article{XLMRoBERTa, title={Unsupervised Cross-lingual Representation Learning at Scale}, author={Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin}, journal={arXiv:1911.02116}, year={2019}}
```
