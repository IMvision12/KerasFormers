# Grounding DINO (open-set object detection)

**Paper**: [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

Grounding DINO detects objects named by a **free-text prompt**: a list of phrases
or a caption: instead of a fixed label set. A Swin image backbone and a BERT text
encoder are fused by a cross-modality encoder, query proposals are picked by
image-text similarity, and boxes are refined in a DINO-style decoder. Each output
box is scored against the prompt tokens, so the "classes" are whatever you ask for.

## Architecture Highlights

- **Dual backbone:** Swin Transformer over the image + a BERT text encoder over the
  prompt; both are kept in-folder (HF-format Swin + BERT), no external backbone deps.
- **Cross-modality encoder:** 6 deformable encoder layers with bi-directional
  text↔image attention fusion, so text features attend to image regions and back.
- **Contrastive query selection:** a two-stage scheme picks the top encoder outputs
  (by image-text similarity) as decoder queries.
- **DINO decoder:** iterative bounding-box refinement + contrastive (token-level)
  classification head, so a box's score is a similarity over the prompt tokens.

## Available Variants

| Variant | Image backbone | HF original |
|---|---|---|
| `grounding_dino_tiny` | Swin-Tiny | `IDEA-Research/grounding-dino-tiny` |
| `grounding_dino_base` | Swin-Base | `IDEA-Research/grounding-dino-base` |

Two classes are exposed:

- `GroundingDinoModel`: backbone + cross-modality encoder/decoder (raw features).
- `GroundingDinoForObjectDetection`: adds the box + contrastive heads (detection).

## Weights

Pre-converted Keras weights are cached from the `grounding_dino` GitHub release on
first use:
[https://github.com/IMvision12/KerasFormers/releases/tag/grounding_dino](https://github.com/IMvision12/KerasFormers/releases/tag/grounding_dino).
You can also convert the original IDEA-Research checkpoints on the fly with
`from_weights("hf:IDEA-Research/grounding-dino-tiny")`.

## Basic Usage

```python
from kerasformers.models.grounding_dino import (
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
)

model = GroundingDinoForObjectDetection.from_weights("grounding_dino_tiny")
processor = GroundingDinoProcessor.from_weights("grounding_dino_tiny")

# or the original checkpoint straight from the Hub
model = GroundingDinoForObjectDetection.from_weights("hf:IDEA-Research/grounding-dino-tiny")

# untrained
model = GroundingDinoForObjectDetection.from_weights("grounding_dino_tiny", load_weights=False)
```

## Inference Example

The prompt is a lower-cased caption with phrases separated by periods (the
Grounding DINO convention, e.g. `"a cat. a remote control."`).

```python
from PIL import Image
from kerasformers.models.grounding_dino import (
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
)

model = GroundingDinoForObjectDetection.from_weights("grounding_dino_tiny")
processor = GroundingDinoProcessor.from_weights("grounding_dino_tiny")

image = Image.open("assets/coco/coco_paddleboard.jpg").convert("RGB")

# Prompts read best without articles: in "a paddle" the "a" can outscore the noun.
inputs = processor(images=image, text=["person", "paddle", "board"])
outputs = model(inputs)

results = processor.post_process_object_detection(
    outputs,
    threshold=0.3,
    target_sizes=[(image.height, image.width)],
    input_ids=inputs["input_ids"],
)[0]

detections = sorted(
    zip(results["scores"], results["text_labels"], results["boxes"]),
    key=lambda d: -float(d[0]),
)
for score, name, box in detections:
    print(f"{name:8s} {float(score):.3f}  {[round(float(v)) for v in box]}")
```

```
person   0.911  [449, 119, 500, 242]
board    0.603  [361, 230, 578, 247]
paddle   0.588  [399, 159, 476, 222]
```

`post_process_object_detection` returns, per image, `scores`, `labels`, and `boxes`
(xyxy pixel coordinates), the same schema as every other detector in the library.
`labels` holds the prompt-**token position** each query matched, which is what the
open-set head predicts. Pass `input_ids` to also get `text_labels`, those positions
decoded back to strings.

A query matches a single token rather than a span, so a multi-word phrase like
"paddle board" is reported by whichever token scored highest. Keep prompts to single
words where you can. Omitting `target_sizes` leaves boxes normalized to `[0, 1]`.

## Parity vs HuggingFace Reference

Validated against `transformers` (latest main) on a real forward pass: backbone
features cosine `1.0` (max|Δ| `2e-5`), detector **boxes 0.0**, class logits cosine
`1.0`, probabilities `5e-5`. Verified on the `torch`, `jax`, and `tensorflow`
backends. Reproduce with:

```bash
KERAS_BACKEND=torch python -m kerasformers.models.grounding_dino.convert_grounding_dino_hf_to_keras
```
