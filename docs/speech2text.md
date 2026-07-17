# Speech2Text (S2T)

**Paper**: [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171)

Speech2Text is Facebook/fairseq's convolution + Transformer encoder-decoder
for end-to-end speech recognition (and, in its multilingual variants, speech
translation). The encoder ingests 80-channel log-mel **filterbank** features
through a 2-layer 1-D convolutional subsampler (kernel 5, stride 2, GLU) that
downsamples the time axis by 4x, adds fixed **sinusoidal** positions, and runs
a stack of pre-LN transformer blocks. The decoder generates SentencePiece token
ids autoregressively, attending to the encoder output via cross-attention, with
a separate linear LM head. Token + conv embeddings are scaled by
`sqrt(d_model)`.

kerasformers ships a **pure Keras 3** port of the three LibriSpeech ASR
checkpoints with bit-close parity to HuggingFace's reference implementation.
The feature extractor, encoder, decoder, and greedy `generate` loop run
unmodified on TensorFlow / Torch / JAX backends: no `transformers` or `torch`
runtime dependency.

## Classes

Two classes are exposed, mirroring HF's `Speech2Text*` hierarchy:

| Class | HF equivalent | Purpose |
|---|---|---|
| `Speech2TextModel` | `Speech2TextModel` / `Speech2TextForConditionalGeneration` | Encoder + decoder + LM head. Functional graph for teacher-forced training and forward passes. |
| `Speech2TextSpeechToText` | `Speech2TextForConditionalGeneration` + `.generate()` | Subclass of `Speech2TextModel` that adds an end-to-end `.generate(audio, processor)` method for transcription. |

Both are loaded the same way:

```python
from kerasformers.models.speech2text import Speech2TextSpeechToText

# kerasformers release variant
model = Speech2TextSpeechToText.from_weights("s2t-small-librispeech-asr")

# Any HF Hub repo whose model_type is "speech_to_text"
model = Speech2TextSpeechToText.from_weights("hf:facebook/s2t-small-librispeech-asr")
```

## Model Variants

| Variant id | Params | Layers (enc / dec) | d_model | Heads | Mel bins | Vocab |
|---|---|---|---|---|---|---|
| `s2t-small-librispeech-asr` | 30 M | 12 / 6 | 256 | 4 | 80 | 10 000 |
| `s2t-medium-librispeech-asr` | 71 M | 12 / 6 | 512 | 8 | 80 | 10 000 |
| `s2t-large-librispeech-asr` | 268 M | 12 / 6 | 1024 | 16 | 80 | 10 000 |

All three are trained on LibriSpeech (English, lowercase) and share the **same
10k SentencePiece vocabulary**: only the model size differs. The conv
subsampler (2x kernel-5 / stride-2 + GLU), `scale_embedding`, ReLU FFNs, and
sinusoidal positions are identical across variants.

## Available Weights

Each variant ships one combined `.weights.h5` file (encoder + decoder + LM
head) converted from the official Facebook checkpoints, hosted under the
kerasformers
[`speech2text`](https://github.com/IMvision12/KerasFormers/releases/tag/speech2text)
release tag and downloaded on first use, then cached locally.

Variant ids for `Speech2TextModel.from_weights`:

| Variant id | Source |
|---|---|
| `s2t-small-librispeech-asr` | `facebook/s2t-small-librispeech-asr` |
| `s2t-medium-librispeech-asr` | `facebook/s2t-medium-librispeech-asr` |
| `s2t-large-librispeech-asr` | `facebook/s2t-large-librispeech-asr` |

## Model

`Speech2TextModel` is a `FunctionalBaseModel` (Functional) subclass that wires the encoder
and decoder into a single graph. Both are exposed as attributes for
inference / generation paths:

```python
from kerasformers.models.speech2text import Speech2TextModel

# kerasformers release variant
model = Speech2TextModel.from_weights("s2t-small-librispeech-asr")

# Any HF Hub repo whose model_type is "speech_to_text"
model = Speech2TextModel.from_weights("hf:facebook/s2t-small-librispeech-asr")

model.encoder        # keras.Model: input_features -> (B, T // 4, hidden_dim)
model.decoder        # keras.Model: {decoder_input_ids, encoder_hidden_states} -> logits
model.hidden_dim     # 256
model.vocab_size     # 10000

# Joint forward pass (teacher-forced training):
out = model({
    "input_features":    fbank,  # (B, T, 80)
    "decoder_input_ids": ids,    # (B, L)
})
out["encoder_hidden_states"]     # (B, T // 4, hidden_dim)
out["logits"]                    # (B, L, vocab_size)
```

> **Note**: unlike Whisper, the fbank features are laid out
> `(B, T, num_mel_bins)` (time first), not `(B, num_mel_bins, T)`.

The class is also constructable directly with custom hyperparameters for
from-scratch training:

```python
from kerasformers.models.speech2text import Speech2TextModel

model = Speech2TextModel(
    hidden_dim=256,
    encoder_num_layers=12, decoder_num_layers=6,
    encoder_attention_heads=4, decoder_attention_heads=4,
    encoder_ffn_dim=2048, decoder_ffn_dim=2048,
    vocab_size=10000, num_mel_bins=80,
    conv_channels=1024, conv_kernel_sizes=(5, 5), num_conv_layers=2,
    scale_embedding=True, activation_function="relu",
)
```

## Loading HF Fine-tunes

Any HF repo whose `model_type` is `"speech_to_text"` can be loaded directly via
`Speech2TextModel.from_weights("hf:<repo>")`: the class reads `d_model`, depth,
head count, conv config, and `scale_embedding` straight from the HF config. The
converter normalizes both `Speech2TextForConditionalGeneration`
(`model.encoder.*` prefix) and `Speech2TextModel` (`encoder.*` prefix)
state-dict layouts, and handles a tied or untied `lm_head`.

## Features and Capabilities

- **End-to-end ASR**: `Speech2TextSpeechToText` extends `Speech2TextModel` and
  adds an `.generate(audio, processor)` method: mirrors HF's
  `Speech2TextForConditionalGeneration` (model class + `.generate()`).
- **Single processor entry point**: `Speech2TextProcessor` bundles the
  feature extractor + tokenizer behind one object matching the HF API surface.
- **Pure Keras 3**: the fbank feature extractor uses `keras.ops` (`rfft`,
  `matmul`, `log`) and runs on any backend; the tokenizer is SentencePiece +
  `vocab.json` (no `transformers`).
- **HF passthrough**: `from_weights("hf:org/repo")` works for the original
  Facebook checkpoints and any community fine-tune whose `model_type` is
  `"speech_to_text"`, including the bare `Speech2TextModel` (no LM head).
- **Fine-tunable**: every variable in the encoder + decoder + LM head is
  trainable.

## Basic Usage

The shortest path is `Speech2TextSpeechToText`: same model graph as
`Speech2TextModel` plus an end-to-end `.generate(audio, processor)` method
(audio in, text out).

```python
from kerasformers.models.speech2text import (
    Speech2TextSpeechToText,
    Speech2TextProcessor,
)

model = Speech2TextSpeechToText.from_weights("s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_weights("s2t-small-librispeech-asr")

# raw_audio: 1-D float32 in [-1, 1] at 16 kHz
text = model.generate(raw_audio, processor)
print(text)        # ['mister quilter is the apostle of the middle classes ...']
```

LibriSpeech S2T outputs are **lowercase, unpunctuated** (that's how the model
was trained).

## Processor

`Speech2TextProcessor` is the recommended top-level entry point: it bundles
the feature extractor and tokenizer behind a single object that mirrors
HuggingFace's `transformers.Speech2TextProcessor` API.

```python
from kerasformers.models.speech2text import Speech2TextProcessor

processor = Speech2TextProcessor.from_weights("s2t-small-librispeech-asr")

# audio path
out = processor(audio=wave, sampling_rate=16000)
# {"input_features": (B, T, 80)}

# label path (fine-tuning)
out = processor(text=["hello world", "foo bar"])
# {"input_ids": (B, L)}

# decoded text
text = processor.decode(ids, skip_special_tokens=True)
texts = processor.batch_decode(ids_batch, skip_special_tokens=True)

# the underlying components are still accessible
processor.feature_extractor   # Speech2TextFeatureExtractor
processor.tokenizer           # Speech2TextTokenizer
processor.decoder_start_token_id  # 2  (</s>, Bart-style seed)
```

## Feature Extractor

`Speech2TextFeatureExtractor` is a **pure Keras 3** Kaldi-style log-mel
filterbank (fbank) extractor: the spectrogram math goes through `keras.ops`
(`rfft`, `matmul`, `log`), so the same code runs on TF / Torch / JAX.

Pipeline (matches the reference Kaldi fbank):

1. Scale the waveform to int16 range (`x * 2**15`).
2. Frame at 25 ms / 10 ms (snip-edges), per-frame DC removal, 0.97
   pre-emphasis, Povey window.
3. 512-point power spectrum → 80-channel HTK-mel filterbank → `log`.
4. Per-utterance cepstral mean-variance normalization (CMVN).

```python
from kerasformers.models.speech2text import Speech2TextFeatureExtractor

feat = Speech2TextFeatureExtractor(sampling_rate=16000, num_mel_bins=80)
fbank = feat(raw_audio_or_list_of_waves)   # (B, T, 80)
```

Verified against HF `Speech2TextFeatureExtractor` to **max diff ~5.4e-5** on
real audio.

## Tokenizer

`Speech2TextTokenizer` is a SentencePiece tokenizer: an SP model turns text
into subword pieces and a separate `vocab.json` maps pieces to ids. It is used
mainly to **decode** generated ids back to text (`ids -> pieces -> SP decode`);
the encode path is provided for label preparation. The LibriSpeech vocabulary
is lowercase and uses `</s>` (id 2) as both the decoder start token and the
end-of-sequence token (Bart convention).

```python
from kerasformers.models.speech2text import Speech2TextTokenizer

tok = Speech2TextTokenizer.from_weights("s2t-small-librispeech-asr")  # downloads vocab.json + spm model
text = tok.decode([10, 42, 2], skip_special_tokens=True)
```

The two files (`vocab.json`, `sentencepiece.bpe.model`) are **shared across all
three variants** and hosted on the kerasformers `speech2text` release tag,
downloaded on first use.

## Generation

The greedy decoding loop is a method on `Speech2TextSpeechToText`. Decoding is
seeded with `decoder_start_token_id` (`</s>` = 2) and stops at the next `</s>`.

```python
text = model.generate(wave, processor, max_new_tokens=200)
ids = model.generate(wave, processor, return_ids=True)   # List[List[int]]
```

`model.encoder(fbank)` and `model.decoder({"decoder_input_ids": ids,
"encoder_hidden_states": enc_out})` are exposed directly for custom decoding
loops (beam search, prefix scoring, KV-cache, etc.).

## Fine-tuning

All variables in the encoder + decoder + LM head are trainable. The processor's
text path feeds the label tensor:

```python
import keras
from kerasformers.models.speech2text import Speech2TextModel, Speech2TextProcessor

model = Speech2TextModel.from_weights("s2t-small-librispeech-asr")
encoder, decoder = model.encoder, model.decoder
processor = Speech2TextProcessor.from_weights("s2t-small-librispeech-asr")

inputs = processor(audio=audio_batch, sampling_rate=16000)  # input_features
labels = processor(text=text_batch)["input_ids"]            # label ids

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# drive encoder -> decoder with a teacher-forced decoder_input_ids and
# optimize against `labels`.
```

## Citation

```bibtex
@inproceedings{wang2020fairseqs2t,
  title={fairseq S2T: Fast Speech-to-Text Modeling with fairseq},
  author={Wang, Changhan and Tang, Yun and Ma, Xutai and Wu, Anne and
          Okhonko, Dmytro and Pino, Juan},
  booktitle={Proceedings of the 2020 Conference of the Asian Chapter of the
             Association for Computational Linguistics (AACL): System
             Demonstrations},
  year={2020}
}
```
