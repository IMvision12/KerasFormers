# Moonshine

**Paper**: [Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608)

Moonshine is Useful Sensors' encoder-decoder transformer for fast, low-latency
automatic speech recognition. Unlike Whisper, the encoder consumes the **raw
16 kHz waveform** directly — a 3-layer Conv1D stem (kernel 127 / stride 64,
no bias → `tanh` → GroupNorm; kernel 7 / stride 3 → GELU; kernel 3 / stride 2 →
GELU) replaces the log-mel front end — followed by a stack of pre-LN transformer
blocks with GLM-style **partial** rotary position embeddings. The decoder
generates BPE token ids autoregressively, attending to the encoder output via
cross-attention; its MLP is a gated-SiLU (`fc1` projects to `2 × ffn`, split into
value/gate). Every LayerNorm is **bias-free** (scale only), and the token
embedding is tied with the LM head.

kerasformers ships a **pure Keras 3** port of both official Useful Sensors
checkpoints with bit-close parity to HuggingFace's reference implementation. The
processor, encoder, decoder, and greedy `generate` loop run unmodified on
TensorFlow / Torch / JAX backends — no `transformers` or `torch` runtime
dependency.

## Classes

Two classes are exposed, mirroring HF's Moonshine hierarchy:

| Class | HF equivalent | Purpose |
|---|---|---|
| `MoonshineModel` | `MoonshineModel` / `MoonshineForConditionalGeneration` | Encoder + decoder + tied LM head. Functional graph for teacher-forced training and forward passes. |
| `MoonshineSpeechToText` | `MoonshineForConditionalGeneration` + `.generate()` | Subclass of `MoonshineModel` that adds an end-to-end `.generate(audio, processor, ...)` method for transcription. |

Both are loaded the same way:

```python
from kerasformers.models.moonshine import MoonshineSpeechToText

# kerasformers release variant
model = MoonshineSpeechToText.from_weights("moonshine_tiny")

# Any HF Hub repo whose model_type is "moonshine"
model = MoonshineSpeechToText.from_weights("hf:UsefulSensors/moonshine-base")
```

## Model Variants

| Variant id | Params | Layers (enc / dec) | hidden | Heads (q / kv) | FFN | Partial RoPE | Vocab |
|---|---|---|---|---|---|---|---|
| `moonshine_tiny` | ~27 M | 6 / 6 | 288 | 8 / 8 | 1152 | 0.9 | 32 768 |
| `moonshine_base` | ~62 M | 8 / 8 | 416 | 8 / 8 | 1664 | 0.62 | 32 768 |

Both use `max_position_embeddings = 194`, `rope_theta = 10000`, GELU encoder MLP,
and a gated-SiLU decoder MLP. The rotary dimension is
`(hidden // heads) × partial_rotary_factor` — only that fraction of each head is
rotated (GLM-style partial RoPE).

## Available Weights

Each variant ships a single `"usefulsensors"` preset converted from the official
Useful Sensors checkpoints on HuggingFace. One combined `.weights.h5` file per
variant (encoder + decoder together — the same file serves both `MoonshineModel`
and `MoonshineSpeechToText`, since the LM head is tied) is hosted under the
kerasformers
[`moonshine`](https://github.com/IMvision12/KerasFormers/releases/tag/moonshine)
release tag and downloaded on first use, then cached locally.

| Variant id | Params | Source |
|---|---|---|
| `moonshine_tiny` | ~27 M | `UsefulSensors/moonshine-tiny` |
| `moonshine_base` | ~62 M | `UsefulSensors/moonshine-base` |

## Model

`MoonshineModel` is a `FunctionalBaseModel` (Functional) subclass that wires the
encoder and decoder into a single graph. Both sub-models are exposed as
attributes for inference / generation paths:

```python
from kerasformers.models.moonshine import MoonshineModel

model = MoonshineModel.from_weights("moonshine_tiny")
model = MoonshineModel.from_weights("hf:UsefulSensors/moonshine-tiny")

model.encoder        # keras.Model: input_values (B, audio_len) -> (B, T, hidden)
model.decoder        # keras.Model: {decoder_input_ids, encoder_hidden_states} -> logits
model.hidden_dim     # 288
model.vocab_size     # 32768

# Joint forward pass (teacher-forced training):
out = model({
    "input_values":      audio,   # (B, audio_length)  raw 16 kHz waveform
    "decoder_input_ids": ids,     # (B, L)
})
out["encoder_hidden_states"]      # (B, T, hidden_dim)
out["logits"]                     # (B, L, vocab_size)
```

The class is also constructable directly with custom hyperparameters for
from-scratch training:

```python
from kerasformers.models.moonshine import MoonshineModel

model = MoonshineModel(
    hidden_dim=288,
    encoder_num_layers=6, decoder_num_layers=6,
    encoder_attention_heads=8, decoder_attention_heads=8,
    encoder_ffn_dim=1152, decoder_ffn_dim=1152,
    vocab_size=32768,
    max_position_embeddings=194,
    partial_rotary_factor=0.9,
    rope_theta=10000.0,
    encoder_activation="gelu",     # encoder MLP
    decoder_activation="silu",     # gated decoder MLP
    layer_norm_eps=1e-5,
)
```

## Loading HF Fine-tunes

Any HF repo whose `model_type` is `"moonshine"` can be loaded directly via
`from_weights("hf:<repo>")` — the class reads hidden size, depth, head counts,
GQA kv-head counts, activations, partial-rotary factor, and rope theta straight
from the HF config (`config_from_hf`), then converts the state dict. This covers
the original Useful Sensors checkpoints and any community fine-tune sharing the
same architecture.

```python
from kerasformers.models.moonshine import MoonshineSpeechToText, MoonshineProcessor

model = MoonshineSpeechToText.from_weights("hf:UsefulSensors/moonshine-base")
processor = MoonshineProcessor.from_hf("UsefulSensors/moonshine-base")
text = model.generate(audio, processor)
```

## Features and Capabilities

- **Raw-waveform input**: the encoder ingests the 16 kHz waveform directly via a
  Conv1D stem — no mel-spectrogram step, fewer ops, lower latency than Whisper.
- **Partial rotary embeddings**: GLM-style RoPE over a fraction of each head
  dimension (`partial_rotary_factor`), applied on self-attention only.
- **GQA-capable**: separate `encoder_num_kv_heads` / `decoder_num_kv_heads`
  (default to the query head counts) for grouped-query attention fine-tunes.
- **Tied LM head**: the decoder token embedding is reused as the output
  projection (`proj_out`), so one weight file covers both classes.
- **Generation in the model class**: `MoonshineSpeechToText` extends
  `MoonshineModel` and adds an end-to-end `.generate(audio, processor, ...)`
  method — mirrors HF's `MoonshineForConditionalGeneration`.
- **Pure Keras 3**: the feature extractor (waveform batching) and the model run
  on any backend; the tokenizer is a Rust-backed `tokenizers.Tokenizer` (no
  `transformers`).
- **HF passthrough**: `from_weights("hf:org/repo")` works for any community
  fine-tune whose `model_type` is `"moonshine"`.
- **Fine-tunable**: every encoder + decoder variable is trainable; gradients flow
  through the tied LM head.

## Basic Usage

The shortest path is `MoonshineSpeechToText` — same model graph as
`MoonshineModel` plus an end-to-end `.generate(audio, processor, ...)` method
(audio in, text out).

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import soundfile as sf
from kerasformers.models.moonshine import MoonshineSpeechToText, MoonshineProcessor

model = MoonshineSpeechToText.from_weights("moonshine_base")
processor = MoonshineProcessor()                  # UsefulSensors moonshine tokenizer

# raw_audio: 1-D float32 in [-1, 1] at 16 kHz
audio, sr = sf.read("assets/librispeech_sample.wav")
assert sr == 16000

text = model.generate(audio, processor, max_new_tokens=200)
print(text[0])
```

`.generate` runs feature extraction (waveform batching), the encoder, greedy
decoding, and detokenization in one call. Pass `return_ids=True` to get raw
token-id lists instead of strings.

### Using the lower-level API directly

For custom decoding (beam search, KV-cache, prefix scoring), call `model.encoder`
and `model.decoder` directly:

```python
inputs = processor(audio=wave, sampling_rate=16000)
enc_out = model.encoder(inputs["input_values"])        # encode once

# then drive decoding however you like — call model.decoder per step with
# {"decoder_input_ids": ids, "encoder_hidden_states": enc_out}
```

## Processor

`MoonshineProcessor` is the recommended top-level entry point — it bundles the
feature extractor and tokenizer and mirrors HF's `MoonshineProcessor` API.

```python
from kerasformers.models.moonshine import MoonshineProcessor

processor = MoonshineProcessor()                                  # default tokenizer
processor = MoonshineProcessor.from_hf("UsefulSensors/moonshine-base")

# audio path
out = processor(audio=wave, sampling_rate=16000)   # {"input_values": (B, audio_len)}

# label path (fine-tuning)
out = processor(text=["hello world", "foo bar"])   # {"input_ids", "attention_mask"}

# decoded text
text = processor.decode(ids, skip_special_tokens=True)
texts = processor.batch_decode(ids_batch, skip_special_tokens=True)

processor.feature_extractor       # MoonshineFeatureExtractor
processor.tokenizer               # MoonshineTokenizer
processor.decoder_start_token_id  # 1  (<s>) — the seed token for generation
```

> The Moonshine `tokenizer.json` is identical across every Useful Sensors
> checkpoint, so the default works for all variants; pass `from_hf(repo)` only if
> you want to pin a specific repo.

## Feature Extractor

`MoonshineFeatureExtractor` is intentionally minimal — a `feature_size = 1`,
`do_normalize = False` extractor that simply stacks a batch of waveforms and
right-zero-pads shorter clips to a common length. There is **no** mel /
spectrogram step (the encoder's Conv1D stem learns the front-end features).

```python
from kerasformers.models.moonshine import MoonshineFeatureExtractor

feat = MoonshineFeatureExtractor(sampling_rate=16000, padding_value=0.0)
values = feat(raw_audio_or_list_of_waves)   # (B, max_audio_len) float32
```

## Tokenizer

`MoonshineTokenizer` wraps a Rust-backed `tokenizers.Tokenizer` (byte-fallback
BPE with a `▁` metaspace normalizer). The `tokenizer.json` is downloaded from the
Useful Sensors Hub repo (`hf_id`, default `UsefulSensors/moonshine-tiny`) — no
runtime `transformers` dependency. Special ids: `<s>` = 1 (bos), `</s>` = 2
(eos), `<unk>` = 0.

```python
from kerasformers.models.moonshine import MoonshineTokenizer

tok = MoonshineTokenizer()                                   # default repo
tok = MoonshineTokenizer(hf_id="UsefulSensors/moonshine-base")
ids = tok.tokenize("Hello, world!")                          # no special tokens added
text = tok.decode(ids, skip_special_tokens=True)
```

The encode path does **not** add special tokens — `MoonshineSpeechToText` seeds
decoding with `decoder_start_token_id` (`<s>`) itself.

## Generation

`MoonshineSpeechToText.generate` is a plain greedy decoding loop (matches the
reference Moonshine generate):

```python
text = model.generate(
    audio, processor,
    max_new_tokens=200,        # decode budget
    sampling_rate=16000,
    return_ids=False,          # True -> List[List[int]]
)
```

Decoding seeds with `<s>`, then argmax-samples each step against the cross-attended
encoder output, stopping when every sequence in the batch has emitted `</s>`.

## Citation

```bibtex
@article{jeffries2024moonshine,
  title={Moonshine: Speech Recognition for Live Transcription and Voice Commands},
  author={Jeffries, Nat and King, Evan and Kudlur, Manjunath and Nicholson, Guy
          and Wang, James and Warden, Pete},
  journal={arXiv preprint arXiv:2410.15608},
  year={2024}
}
```
