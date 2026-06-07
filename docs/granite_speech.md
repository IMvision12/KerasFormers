# Granite Speech

**Paper**: [Granite-speech: open-source speech-aware LLMs with strong English ASR capabilities](https://arxiv.org/abs/2505.08699)

Granite Speech is IBM's speech-aware large language model. A **conformer CTC audio
encoder** and a **BLIP-2 Q-Former projector** turn mel features into audio
embeddings, which are scattered into the `<|audio|>` placeholder positions of a
**Granite text decoder** — exactly the way a vision-language model splices image
embeddings into the token stream. In speech mode the encoder, projector, and a
query/value **LoRA adapter** on the decoder are active; in text mode it is the
plain Granite decoder. The decoder carries Granite's scalar multipliers
(`embedding_multiplier`, `residual_multiplier`, `attention_multiplier`) and a
`logits / logits_scaling` output.

kerasformers ships a **self-contained, pure Keras 3** port: the conformer
encoder, the Q-Former, the inline Granite decoder, and the audio LoRA all live in
the `granite_speech` package — no separate LLM dependency. The forward runs
eagerly through `keras.ops` on TensorFlow / Torch / JAX, validated to **cosine
1.0 / max|diff| ~1e-7** against HF `GraniteSpeechForConditionalGeneration`.

## Classes

| Class | Base | Purpose |
|---|---|---|
| `GraniteSpeechModel` | `SubclassedBaseModel` | Encoder + projector + Granite decoder fused at audio placeholders. Returns `last_hidden_state` (no LM head). |
| `GraniteSpeechGenerate` | `GraniteSpeechModel`, `BaseGeneration` | Adds the (tied) LM head + a fast KV-cache `.generate()` (audio + text → text). |
| `GraniteSpeechTextModel` | `keras.layers.Layer` | The Granite causal decoder itself (`embed → N decoder layers → RMSNorm`); its token embedding is tied as the LM head. |

Loading (release variant or any HF repo whose `model_type` is `"granite_speech"`):

```python
from kerasformers.models.granite_speech import GraniteSpeechGenerate

model = GraniteSpeechGenerate.from_weights("granite_speech_3_3_2b")
model = GraniteSpeechGenerate.from_weights("hf:ibm-granite/granite-speech-3.3-2b")
```

## Model Variants

| Variant id | Params | LLM (layers / hidden / heads q·kv) | Vocab | Encoder (conformer) | Projector (Q-Former) |
|---|---|---|---|---|---|
| `granite_speech_3_3_2b` | ~2.5 B | 40 / 2048 / 32·8 | 49 160 | 16 layers, hidden 1024, out 256 | 1024 hidden, 2 layers, window 15 / downsample 5 |

LLM specifics: `rope_theta = 1e7`, `embedding_multiplier = 12.0`,
`residual_multiplier = 0.22`, `attention_multiplier = 1/64`, `logits_scaling = 8`,
tied embeddings, audio token id `49159`, and a rank-64 query/value LoRA adapter
(`has_lora_adapter = True`) enabled only when audio is present. The larger
`ibm-granite/granite-speech-3.3-8b` shares the architecture and can be loaded via
`from_weights("hf:ibm-granite/granite-speech-3.3-8b")`.

> **Granite Speech Plus** (`granite-speech-4.1-2b-plus`) is the granite-4.0-based
> successor — same architecture, different config + tokenizer. See
> [`granite_speech_plus.md`](granite_speech_plus.md).

## Available Weights

Weights are **sharded** (the 2 B checkpoint exceeds a single `.weights.h5`): a
`granite_speech_3_3_2b.weights.json` index plus `_NNNNN.weights.h5` shards, hosted
under the kerasformers
[`granite_speech`](https://github.com/IMvision12/KerasFormers/releases/tag/granite_speech)
release tag and downloaded on first use. Because the model is subclassed (built
lazily), `from_release` does a dummy forward to build the graph **before** loading
the shards — handled automatically inside `from_weights` / `from_release`.

## Model

`GraniteSpeechModel` is the backbone (no LM head). It accepts a dict with one
text tensor and optional audio tensors; the audio keys can be omitted entirely for
text-only use.

```python
from kerasformers.models.granite_speech import GraniteSpeechModel

model = GraniteSpeechModel.from_weights("granite_speech_3_3_2b")

out = model({
    "input_ids":           input_ids,            # (B, L) int — contains <|audio|> ids
    "input_features":      input_features,        # (num_audios, frames, 160) mel
    "input_features_mask": input_features_mask,   # (num_audios, max_proj_len) bool, optional
})
out["last_hidden_state"]   # (B, L, 2048)

# text-only
out = model({"input_ids": input_ids, "attention_mask": attention_mask})
```

Internally: the CTC encoder + Q-Former produce audio embeddings
(`get_audio_features`), `merge_audio_embeddings` scatters them into the
`<|audio|>` slots, the embeddings are scaled by `embedding_multiplier`, and the
Granite decoder runs with the LoRA adapter enabled iff audio was supplied.

`GraniteSpeechGenerate` adds the tied vocabulary projection (`logits /
logits_scaling`) and returns both tensors:

```python
out = GraniteSpeechGenerate.from_weights("granite_speech_3_3_2b")({...})
out["logits"]              # (B, L, vocab_size)
out["last_hidden_state"]   # (B, L, embed_dim)
```

## Basic Usage

The processor builds every tensor the model needs; `GraniteSpeechGenerate.generate`
runs the rest. Audio + text in, text out.

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import soundfile as sf
from kerasformers.models.granite_speech import (
    GraniteSpeechGenerate, GraniteSpeechProcessor,
)

model = GraniteSpeechGenerate.from_weights("granite_speech_3_3_2b")
processor = GraniteSpeechProcessor()

audio, sr = sf.read("assets/sample.wav")     # 16 kHz mono float32
assert sr == 16000

conversation = [
    {"role": "user", "content": [
        {"type": "audio"},
        {"type": "text", "text": "Can you transcribe this audio?"},
    ]},
]

inputs = processor(conversation=conversation, audio=audio)
output_ids = model.generate(**inputs, max_new_tokens=200)
print(processor.tokenizer.decode(output_ids[0]))
```

`processor(...)` returns `input_ids` / `attention_mask` plus `input_features` /
`input_features_mask`; `generate(**inputs, ...)` forwards the audio tensors to the
prefill step via `**prefill_inputs`.

## Generation

`.generate` comes from `BaseGeneration` and is backend-compiled (jax.jit /
`tf.function(jit_compile=True)` / torch eager):

```python
output_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    input_features=input_features,            # forwarded to the prefill cache
    input_features_mask=input_features_mask,
    max_new_tokens=200,
)
```

`build_cache` runs the audio encoder + projector + audio-token splice **once**
into a fixed KV cache (LoRA enabled because audio is present), then
`call_with_cache` performs text-only decode steps (LoRA stays enabled for the
whole turn). The default `eos_token_id` is `0`.

## Processor

`GraniteSpeechProcessor` bundles the feature extractor and tokenizer and handles
the `<|audio|>` placeholder expansion.

```python
from kerasformers.models.granite_speech import GraniteSpeechProcessor

processor = GraniteSpeechProcessor()                      # release tokenizer
processor = GraniteSpeechProcessor.from_hf("ibm-granite/granite-speech-3.3-2b")

# chat-style (audio + text)
inputs = processor(conversation=conversation, audio=audio)

# raw text + audio
inputs = processor(text="<|audio|> transcribe this", audio=audio)

processor.feature_extractor   # GraniteSpeechFeatureExtractor
processor.tokenizer           # GraniteSpeechTokenizer
```

Each `<|audio|>` placeholder is expanded to the projector output length for its
clip (`audio_embed_sizes`) before tokenization, so the count of `<|audio|>` token
ids in `input_ids` matches the number of audio embeddings the model splices in.

## Feature Extractor

`GraniteSpeechFeatureExtractor` is a **pure Keras 3** mel-spectrogram extractor
(torchaudio-style `MelSpectrogram`: `n_fft = 512`, win 400, hop 160, 80 HTK mel
bins, power spectrogram, centered reflect pad), followed by `log10`, a `max - 8.0`
clamp, `/4 + 1`, and a pair-stacking of consecutive frames → `input_features` of
width `2 × n_mels = 160`. It also returns `audio_embed_sizes` and a boolean
`input_features_mask` over the padded projector tokens.

```python
from kerasformers.models.granite_speech import GraniteSpeechFeatureExtractor

feat = GraniteSpeechFeatureExtractor()
out = feat(audio, sampling_rate=16000)
out["input_features"]        # (num_audios, frames, 160)
out["input_features_mask"]   # (num_audios, max_proj_len) bool
out["audio_embed_sizes"]     # projector output length per clip
```

## Tokenizer

`GraniteSpeechTokenizer` is the Granite BPE tokenizer (Rust-backed
`tokenizers.Tokenizer`) with the `<|audio|>` special token added. The
`tokenizer.json` is downloaded from the kerasformers `granite_speech` release tag
(`GRANITE_SPEECH_TOKENIZER_URL`); `from_hf(repo)` is the opt-in path to fetch it
from an HF repo instead.

```python
from kerasformers.models.granite_speech import GraniteSpeechTokenizer

tok = GraniteSpeechTokenizer()                                       # release file
tok = GraniteSpeechTokenizer.from_hf("ibm-granite/granite-speech-3.3-2b")
ids = tok.encode("hello world")
text = tok.decode(ids)
tok.audio_token, tok.audio_token_id     # "<|audio|>", 49159
```

## Citation

```bibtex
@article{saon2025granitespeech,
  title={Granite-speech: open-source speech-aware LLMs with strong English ASR capabilities},
  author={Saon, George and others},
  journal={arXiv preprint arXiv:2505.08699},
  year={2025}
}
```
