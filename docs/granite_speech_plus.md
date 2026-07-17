# Granite Speech Plus

**Model card**: [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) · **Architecture paper**: [Granite-speech](https://arxiv.org/abs/2505.08699)

Granite Speech Plus is the **granite-4.0-based** successor to Granite Speech. The
architecture is identical: a conformer CTC encoder + BLIP-2 Q-Former projector +
Granite decoder, fused at `<|audio|>` positions, so the kerasformers port simply
**reuses the `granite_speech` implementation**; only the config and tokenizer
differ. Read [`granite_speech.md`](granite_speech.md) first; this page documents
just the deltas.

## What's different from Granite Speech 3.3

| | `granite_speech_3_3_2b` | `granite_speech_4_1_2b_plus` |
|---|---|---|
| LLM base | Granite 3.3 | **Granite 4.0** |
| Vocab | 49 160 | **100 353** |
| MLP dim | 8192 | **4096** |
| Heads (q / kv) | 32 / 8 | **16 / 4** |
| `rope_theta` | 1e7 | **10000** |
| `attention_multiplier` | 1/64 | **1/128** |
| `audio_token_id` | 49159 | **100352** |
| LoRA adapter | yes (rank 64) | **none** (`has_lora_adapter = False`: weights fully merged) |
| `cat_hidden_layers` | `None` | **`[3]`** |
| Encoder `output_dim` | 256 | **348** |
| Tokenizer | `granite_speech_tokenizer.json` | **`granite_speech_plus_tokenizer.json`** (granite-4.0 vocab, `<think>`/`<tool_call>` tokens) |

`hidden_size` (2048) and `num_layers` (40) are unchanged.

**`cat_hidden_layers = [3]`** is the one structural change: the CTC encoder
concatenates its layer-3 intermediate output with the final output before the
projector, so the projector's `encoder_hidden_size` becomes
`encoder_hidden_dim × (len(cat_hidden_layers) + 1) = 1024 × 2 = 2048`
(derived automatically by the model).

## Classes

Thin subclasses that only swap the config / tokenizer source: everything else is
inherited from `granite_speech`:

| Class | Inherits | Note |
|---|---|---|
| `GraniteSpeechPlusModel` | `GraniteSpeechModel` | `BASE_MODEL_CONFIG` → plus config; `HF_MODEL_TYPE = "granite_speech_plus"` |
| `GraniteSpeechPlusGenerate` | `GraniteSpeechGenerate` | LM head + fast `.generate()` |
| `GraniteSpeechPlusTokenizer` | `GraniteSpeechTokenizer` | overrides `TOKENIZER_URL` → the plus `tokenizer.json` |
| `GraniteSpeechPlusProcessor` | `GraniteSpeechProcessor` | overrides `TOKENIZER_CLS` → `GraniteSpeechPlusTokenizer` |

The mel `GraniteSpeechFeatureExtractor` is shared unchanged (re-exported from the
plus package).

## Model Variants

| Variant id | Params | LLM (layers / hidden / heads q·kv) | Vocab |
|---|---|---|---|
| `granite_speech_4_1_2b_plus` | ~2.5 B | 40 / 2048 / 16·4 | 100 353 |

## Available Weights

Sharded, like Granite Speech: a `granite_speech_4_1_2b_plus.weights.json` index +
`_NNNNN.weights.h5` shards under the kerasformers
[`granite_speech`](https://github.com/IMvision12/KerasFormers/releases/tag/granite_speech)
release tag. The plus `tokenizer.json` is hosted on the same tag as
`granite_speech_plus_tokenizer.json`.

## Usage

Identical to Granite Speech: just use the `Plus` classes (the processor builds
the right tensors and the plus tokenizer is wired automatically):

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import soundfile as sf
from kerasformers.models.granite_speech_plus import (
    GraniteSpeechPlusGenerate, GraniteSpeechPlusProcessor,
)

model = GraniteSpeechPlusGenerate.from_weights("granite_speech_4_1_2b_plus")
# or: GraniteSpeechPlusGenerate.from_weights("hf:ibm-granite/granite-speech-4.1-2b-plus")
processor = GraniteSpeechPlusProcessor()      # downloads the granite-4.0 tokenizer

audio, sr = sf.read("assets/sample.wav")      # 16 kHz mono float32
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

See [`granite_speech.md`](granite_speech.md) for the model forward signature,
feature-extractor details, generation internals, and the conformer/Q-Former
architecture: all shared.

## Citation

```bibtex
@article{saon2025granitespeech,
  title={Granite-speech: open-source speech-aware LLMs with strong English ASR capabilities},
  author={Saon, George and others},
  journal={arXiv preprint arXiv:2505.08699},
  year={2025}
}
```
