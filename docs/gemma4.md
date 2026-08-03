# Gemma 4

<div style="background:#fdecea; border:1px solid #f5c6c0; border-radius:3px; padding:12px 16px; color:#4a2626;">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>kerasformers/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/" style="color:#1a5c8a;">Loading Weights</a>.
</div>

<br>

The fourth Gemma generation, ported to pure Keras 3, spanning dense and
Mixture-of-Experts text models plus vision and audio towers, with a 256K context
window. A single `Gemma4Generate` drives generation for both text-only and
multimodal checkpoints, like transformers' `Gemma4ForConditionalGeneration`. The
backbones are `Gemma4Model` (text decoder) and `Gemma4MultimodalModel` (vision +
audio + text).

What changed from Gemma 3:

- **Plain-weight RMSNorm**: the norm weight is used directly, without Gemma's usual
  `1 + w` offset.
- **Zero-pad partial rope**: only `partial_rotary_factor` (0.25) of each head is
  rotated; the remainder is passed through.
- **K=V global MQA**: global layers share a single key/value head
  (`num_global_kv_heads=1`) at a wider `global_head_dim` (512), with key and value
  projections tied (`k_eq_v=True`).
- **Parallel MoE**: the MoE variant runs experts alongside the dense MLP rather than
  replacing it.
- **NaViT vision tower**: patch soft tokens from a 2-D-rotary ViT are scattered onto
  the image placeholders and attend bidirectionally within each image block on the
  sliding-window layers (global layers stay causal).
- **USM audio tower**: a chunked-local-attention Conformer turns audio frames into
  soft tokens, scattered onto the audio placeholders.

Links:

- Paper: [Gemma 4 Technical Report (arXiv:2607.02770)](https://arxiv.org/abs/2607.02770)
- HF docs: [transformers/model_doc/gemma4](https://huggingface.co/docs/transformers/model_doc/gemma4)

See also [gemma.md](gemma.md), [gemma2.md](gemma2.md), [gemma3.md](gemma3.md).

## Variants

Load any of these with `from_weights("<variant>")`. The `-it` suffix marks
instruction-tuned checkpoints (use the chat template).

| Variant | Hub | Architecture | Vision |
|---|---|---|---|
| `gemma-4-12b` | [`google/gemma-4-12B`](https://huggingface.co/google/gemma-4-12B) | dense | text only here (see note) |
| `gemma-4-12b-it` | [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) | dense | text only here (see note) |
| `gemma-4-31b-it` | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) | dense | yes |
| `gemma-4-26b-a4b-it` | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) | MoE, 26B total / 4B active | yes |

Load any variant with `Gemma4Generate`: it builds the vision tower automatically
for `31b-it` and `26b-a4b-it` (which carry one, no audio tower) and stays text-only
for the `12b` checkpoints.

Note that MoE memory is governed by **total** parameters, not active ones: every
expert is resident, so `gemma-4-26b-a4b-it` needs room for 26B weights even though
only 4B are used per token.

**Architecture note.** The `31B` and `26B-A4B` checkpoints are `model_type`
`gemma4`: their vision tower is ported here. The `12B` checkpoints are the separate
`gemma4_unified` architecture, whose vision/audio towers differ and are not ported;
`Gemma4Generate` still loads their (identical) text weights. The audio tower is
ported (validated against the `gemma4_audio` conformer) but the checkpoints that
carry it are the E-variants, whose per-layer-input text decoder is not supported, so
there is no audio-carrying variant wired for Hub / bare-variant loading yet.

## API

### `Gemma4Model`

The decoder backbone, no LM head. Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `262144` | token vocabulary size |
| `embed_dim` | `3840` | model width |
| `mlp_dim` | `15360` | GeGLU inner width |
| `num_layers` | `48` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `8` | key/value heads on local layers (GQA) |
| `num_global_kv_heads` | `1` | key/value heads on global layers (MQA) |
| `head_dim` | `256` | per-head width on local layers |
| `global_head_dim` | `512` | per-head width on global layers |
| `k_eq_v` | `True` | tie the key and value projections on global layers |
| `sliding_window` | `1024` | local attention span |
| `sliding_window_pattern` | `6` | one global layer every N |
| `partial_rotary_factor` | `0.25` | fraction of each head that gets rotated |
| `final_logit_softcapping` | `30.0` | tanh cap on output logits |
| `enable_moe` | `False` | turn on the parallel MoE block |
| `num_experts` | `0` | expert count when `enable_moe` |
| `num_experts_per_tok` | `0` | experts routed per token |
| `moe_mlp_dim` | `0` | per-expert inner width |

### `Gemma4Generate`

The single generation class, over the `Gemma4MultimodalModel` backbone plus a
(tied) LM head with final-logit softcapping (like transformers'
`Gemma4ForConditionalGeneration`). Returns
`{"logits": (batch, seq, vocab_size)}` and adds `.generate()`. It handles both
regimes: text-only when the checkpoint has no towers (or when no image/audio tensors
are passed), and multimodal when it does. The multimodal prefill fuses the soft
tokens and applies the blockwise vision mask; decoding is always text-only over the
per-layer KV cache. Constructor args match `Gemma4MultimodalModel` below.

```python
generate(
    input_ids,
    attention_mask=None,
    max_new_tokens=None,
    eos_token_id=None,
    sampler=None,
    seed=None,
    pixel_values=None,
    pixel_position_ids=None,
    input_features=None,
    input_features_mask=None,
)
```

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | `(batch, seq)` token ids |
| `attention_mask` | `None` | `(batch, seq)` 1 = keep, 0 = padding |
| `max_new_tokens` | `None` | tokens to generate |
| `eos_token_id` | `None` | stop token (defaults to the tokenizer's) |
| `sampler` | `None` | sampling strategy; greedy when unset |
| `seed` | `None` | seed for stochastic samplers |
| `pixel_values` / `pixel_position_ids` | `None` | image patches + 2-D coords (multimodal only) |
| `input_features` / `input_features_mask` | `None` | audio frames + mask (multimodal only) |

### `Gemma4MultimodalModel`

The multimodal backbone: the `Gemma4Model` text decoder plus the vision tower
(`Gemma4VisionModel`) and, when the checkpoint has one, the audio tower
(`Gemma4AudioModel`), with the soft-token projectors. Returns
`{"last_hidden_state": (batch, seq, embed_dim)}`. Image and audio soft tokens are
scattered onto their placeholder positions in `input_ids` before the decoder runs.

| Arg | Default | Meaning |
|---|---|---|
| `text_config` | `None` | dict of `Gemma4Model` constructor args |
| `vision_config` | `None` | dict of `Gemma4VisionModel` constructor args |
| `audio_config` | `None` | dict of `Gemma4AudioModel` args; `None` skips the audio tower |
| `image_token_id` | `258880` | placeholder id filled with image soft tokens |
| `video_token_id` | `258884` | placeholder id filled with video soft tokens |
| `audio_token_id` | `258881` | placeholder id filled with audio soft tokens |
| `pad_token_id` | `0` | id used to embed placeholder slots before the scatter |
| `use_bidirectional_vision` | `True` | blockwise bidirectional attention within each image block |

Call it with a dict:

```python
outputs = model(
    {
        "input_ids": input_ids,  # (batch, seq), image_token_id at image slots
        "pixel_values": pixel_values,  # (batch, num_patches, 3 * patch_size**2)
        "pixel_position_ids": pixel_position_ids,  # (batch, num_patches, 2), (x, y); (-1, -1) = pad
    }
)
```

### `Gemma4VisionModel` and `Gemma4AudioModel`

The towers can be used on their own. `Gemma4VisionModel` takes
`(pixel_values, pixel_position_ids)` and returns pooled soft tokens;
`Gemma4AudioModel` takes `(input_features, input_features_mask)` and returns
`(soft_tokens, valid_mask)`. Both are exposed from
`kerasformers.models.gemma4` for building custom pipelines.

### `Gemma4Tokenizer`

SentencePiece-BPE tokenizer on the `tokenizers` backend.

```python
Gemma4Tokenizer(hf_id=None, tokenizer_file=None)
```

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull `tokenizer.json` from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

Calling it returns `{"input_ids", "attention_mask"}`, padded across the batch. It
accepts a plain string, a list of strings (a batch), or a chat-message list, which is
routed through `apply_chat_template` automatically.

## End-to-end example

### Single input

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from kerasformers.models.gemma4 import Gemma4Generate, Gemma4Tokenizer

model = Gemma4Generate.from_weights("gemma-4-12b-it")
tokenizer = Gemma4Tokenizer.from_weights("gemma-4-12b-it")

inputs = tokenizer(
    [{"role": "user", "content": "Explain rotary embeddings in one sentence."}]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(tokenizer.decode(outputs[0]))
```

### Batch

```python
prompts = [
    "The capital of France is",
    "In one sentence, what is a transformer?",
    "Write a haiku about GPUs.",
]
inputs = tokenizer(prompts)  # {"input_ids": (3, seq), "attention_mask": (3, seq)}
outputs = model.generate(**inputs, max_new_tokens=64)

for text in tokenizer.batch_decode(outputs):
    print(text)
```

### MoE variant

The MoE checkpoint uses the same API; routing is internal:

```python
model = Gemma4Generate.from_weights("gemma-4-26b-a4b-it")
tokenizer = Gemma4Tokenizer.from_weights("gemma-4-26b-a4b-it")
```

### Backbone only

```python
from kerasformers.models.gemma4 import Gemma4Model

backbone = Gemma4Model.from_weights("gemma-4-12b")
hidden = backbone(inputs)["last_hidden_state"]  # (batch, seq, 3840)
```

### Multimodal (vision)

The vision-carrying checkpoints load through the same `Gemma4Generate`. The vision
tensors (`pixel_values` as flattened patches, `pixel_position_ids` as `(x, y)` patch
coordinates) are passed as keyword prefill inputs alongside an `input_ids` sequence
whose image slots hold `image_token_id`:

```python
from kerasformers.models.gemma4 import Gemma4Generate, Gemma4Tokenizer

model = Gemma4Generate.from_weights("gemma-4-31b-it")

outputs = model.generate(
    input_ids,  # image_token_id (258880) at each image slot
    pixel_values=pixel_values,  # (batch, num_patches, 3 * patch_size**2)
    pixel_position_ids=pixel_position_ids,
    max_new_tokens=64,
)
```

A NaViT image processor that packs raw images into `pixel_values` and 2-D patch
coordinates is not shipped yet, so build those tensors yourself (or through the
`transformers` processor) for now.

### Loading from the Hub

```python
model = Gemma4Generate.from_weights("hf:google/gemma-4-12B-it")
model = Gemma4Generate.from_weights("hf:google/gemma-4-31B-it")
```

### Lower memory

The 31B dense and 26B MoE checkpoints need quantization to fit on a single 80GB GPU at
full precision. See [quantization.md](quantization.md):

```python
model = Gemma4Generate.from_weights(
    "gemma-4-31b-it", quantization="int8", low_memory=True, load_dtype="bfloat16"
)
```
