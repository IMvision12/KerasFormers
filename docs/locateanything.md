# LocateAnything (vision-language visual grounding)

**Paper**: [LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding](https://research.nvidia.com/labs/lpr/locate-anything/) (Wang et al., NVIDIA, 2026)

LocateAnything-3B is a generative vision-language model for **visual grounding**: it
takes an image + a natural-language instruction and emits structured coordinates —
boxes, points, and `<ref>` labels — for the requested objects/regions. One model and
one set of weights cover object detection, referring, pointing, OCR/text detection,
and layout/region grounding, selected by the prompt.

## Architecture Highlights

- **MoonViT-SO-400M vision encoder:** a NaViT-packed ViT (native-resolution,
  variable-length block-diagonal attention) with complex 2-D RoPE and a 2×2 patch
  merge, feeding a 2-layer MLP projector (`mlp1`).
- **Qwen2.5-3B-Instruct decoder:** the projected vision tokens are spliced into the
  token-embedding stream at the image-token positions, then run through the reused
  Qwen2 decoder.
- **Parallel Box Decoding (PBD):** a multi-token-prediction head + "magi" block
  attention let the model emit whole boxes/points in parallel for speed.
- **Coordinates as tokens:** boxes are `<box><x1><y1><x2><y2></box>`, points are
  `<box><x><y></box>`, with each coordinate a special token in the `[0, 1000]` grid;
  open-vocabulary outputs are named with `<ref>label</ref>`.

## Available Variants

| Variant | Vision | LLM | HF original |
|---|---|---|---|
| `locateanything_3b` | MoonViT-SO-400M | Qwen2.5-3B-Instruct | `nvidia/LocateAnything-3B` |

Classes: `LocateAnythingModel` (features) · `LocateAnythingGenerate` (+ LM head +
`.generate()`) · `LocateAnythingProcessor` / `LocateAnythingTokenizer` · the helper
`locate_prompt` and the parsers `parse_boxes` / `parse_points` / `parse_grounding`.

## Weights

Pre-converted Keras weights + `tokenizer.json` are cached from the `locateanything`
GitHub release on first use
([releases/tag/locateanything](https://github.com/IMvision12/KerasFormers/releases/tag/locateanything)).
The original checkpoint also converts on the fly with
`from_weights("hf:nvidia/LocateAnything-3B")`.

## Tasks

`locate_prompt(task, text)` builds the verbatim instruction the model was trained on
(the strings match NVIDIA's `LocateAnythingWorker`):

| Task | `locate_prompt(...)` | Parse with |
|---|---|---|
| Object detection | `("detection", ["cat", "car"])` | `parse_boxes` |
| Multi-instance referring | `("referring", "people wearing hats")` | `parse_grounding` |
| Single-instance grounding | `("phrase_grounding", "the blue mug")` | `parse_boxes` |
| Pointing | `("pointing", "the traffic light")` | `parse_points` |
| OCR — scene text | `("ocr")` | `parse_grounding` |
| OCR — text grounding | `("text_grounding", "the total due")` | `parse_boxes` |
| Layout / region | `("layout", "the title bar")` | `parse_grounding` |

For `detection`, pass a **list** of categories (joined with the official `</c>`
separator) or a pre-joined string.

## Basic Usage

```python
from kerasformers.models.locateanything import (
    LocateAnythingGenerate,
    LocateAnythingProcessor,
)

model = LocateAnythingGenerate.from_weights("locateanything_3b")
processor = LocateAnythingProcessor.from_weights("locateanything_3b")

# original checkpoint on the fly
model = LocateAnythingGenerate.from_weights("hf:nvidia/LocateAnything-3B")

# untrained
model = LocateAnythingGenerate.from_weights("locateanything_3b", load_weights=False)
```

## Inference Example

```python
import keras
from PIL import Image
from kerasformers.models.locateanything import (
    LocateAnythingGenerate,
    LocateAnythingProcessor,
    locate_prompt,
)

model = LocateAnythingGenerate.from_weights("locateanything_3b")
processor = LocateAnythingProcessor.from_weights("locateanything_3b")

img = Image.open("image.jpg").convert("RGB")
W, H = img.size

messages = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": locate_prompt("detection", ["cat", "remote"])},
]}]
inputs = processor(conversation=messages)

ids = model.generate(
    keras.ops.convert_to_numpy(inputs["input_ids"]),
    pixel_values=keras.ops.convert_to_numpy(inputs["pixel_values"]),
    image_grid_hws=keras.ops.convert_to_numpy(inputs["image_grid_hws"]),
    tokenizer=processor.tokenizer,
    max_new_tokens=512,
)

for item in processor.parse_grounding(ids[0]):
    # {"label": "cat", "box": [x1, y1, x2, y2]}  — coords in the [0, 1000] grid
    if "box" in item:
        x1, y1, x2, y2 = [c / 1000 for c in item["box"]]
        print(item["label"], [x1 * W, y1 * H, x2 * W, y2 * H])
    elif "point" in item:
        x, y = item["point"][0] / 1000 * W, item["point"][1] / 1000 * H
        print(item["label"], (x, y))
```

Coordinates are returned in the normalized `[0, 1000]` grid; multiply by `W / 1000`
and `H / 1000` for pixels. `parse_boxes` returns `[[x1, y1, x2, y2], …]`,
`parse_points` returns `[[x, y], …]`, and `parse_grounding` pairs each `<ref>` label
with the boxes/points that follow it.

## Generation Modes

`generation_mode` selects the decode path (KV-cached in all three):

- `"hybrid"` (**default**) — Parallel Box Decoding with an autoregressive fallback;
  fastest.
- `"fast"` — MTP only.
- `"slow"` — pure autoregressive.

All three produce clean, correct output. The official stochastic settings are
available too: pass `temperature=0.7, top_p=0.9, repetition_penalty=1.1` to
`.generate()`.

## Parity vs Reference

The MoonViT vision encoder matches NVIDIA's `modeling_vit.py` at cosine `1.0`, the
box-decoding logic is bit-identical to the official `generate_utils`, and end-to-end
the model reproduces the reference's documented boxes (e.g. the COCO cats image: two
cats + two remotes with a clean stop). The full HuggingFace model itself is not used
as a parity reference because NVIDIA's `trust_remote_code` modeling has
transformers-version drift; the kerasformers port loads the safetensors directly.
