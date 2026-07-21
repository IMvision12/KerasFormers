# Main Classes

Every model in KerasFormers is assembled from the same small set of base classes in
`kerasformers.base`. You rarely instantiate them directly, but knowing what they provide
explains why every model page looks alike: the same `from_weights`, the same processor
call, the same `generate`.

```python
from kerasformers.base import (
    FunctionalBaseModel, SubclassedBaseModel,
    BaseGeneration, BaseSeq2SeqGeneration,
    BaseTokenizer, BaseImageProcessor, BaseAudioFeatureExtractor, BaseProcessor,
    Quantizer, fused_attention,
)
```

## Models

Two model bases, differing only in how the graph is built.

### FunctionalBaseModel

```python
class FunctionalBaseModel(keras.Model)
```

Base for models built as a **Keras functional graph**: CLIP, ViT, the detectors,
segmenters, and depth estimators. Because the graph is traced at construction, the input
shape is baked in, which is why these classes take an `image_size` and why changing it
means rebuilding the model.

### SubclassedBaseModel

```python
class SubclassedBaseModel(keras.Model)
```

Base for **imperative** models, mainly the LLMs and VLMs, where the forward pass is
written as code. These accept varying shapes at call time, which is what lets a language
model take any sequence length.

Both share the loading interface below.

## Loading Weights

Every model and preprocessor inherits these three classmethods. What each source actually
does, release variants, `hf:` on-the-fly conversion, and caching, is covered in
[Loading Weights](loading_weights.md).

### from_weights

```python
Model.from_weights(identifier, load_weights=True, skip_mismatch=False,
                   attn_implementation=None, quantization=None, low_memory=False,
                   load_dtype=None, cache_converted=False, low_disk=False, **kwargs)
```

The one entry point you normally use. It dispatches on `identifier`: a bare string is a
kerasformers release variant, and an `hf:`-prefixed string is a Hugging Face repo.

**Parameters**

- **identifier** (`str`): a release variant (`"segformer_b0_ade_512"`) or a Hub repo (`"hf:nvidia/segformer-b0-finetuned-ade-512-512"`).
- **load_weights** (`bool`, *optional*, defaults to `True`): set `False` to build the architecture with random initialization.
- **skip_mismatch** (`bool`, *optional*, defaults to `False`): skip weights whose shapes disagree instead of raising, for partially compatible fine-tunes.
- **attn_implementation** (`str`, *optional*): attention kernel to use, see [`fused_attention`](#fused_attention).
- **quantization** (`str`, *optional*): quantize while loading, for example `"int8"`. See [Quantization](quantization.md).
- **low_memory** (`bool`, *optional*, defaults to `False`): stream weights in rather than materializing the full state dict.
- **load_dtype** (`str`, *optional*): cast weights on load, typically `"bfloat16"`.
- **cache_converted** (`bool`, *optional*, defaults to `False`): keep the converted Keras weights so the next `hf:` load skips conversion.
- **low_disk** (`bool`, *optional*, defaults to `False`): stream shards and evict them as it goes, for checkpoints larger than free disk.
- **kwargs**: forwarded to the constructor, so `image_size=448` or `as_backbone=True` go here.

```python
model = SegFormerSemanticSegment.from_weights("segformer_b0_ade_512")
model = SegFormerSemanticSegment.from_weights("hf:<user>/my-finetune")
model = Qwen3Generate.from_weights("qwen3_8b", load_dtype="bfloat16", low_memory=True)
```

### from_release and from_hf

```python
Model.from_release(variant, load_weights=True, skip_mismatch=False, quantization=None,
                   low_memory=False, low_disk=False, **kwargs)

Model.from_hf(hf_id, load_weights=True, variant=None, skip_mismatch=False,
              quantization=None, low_memory=False, low_disk=False, **kwargs)
```

The two halves `from_weights` dispatches to. Call them directly only when you want to be
explicit about the source. `from_hf` reads the repo's `config.json`, so a fine-tune with a
different class count or vocabulary needs no extra arguments.

> **Load the processor from the same source as the model.** A fine-tune can ship a
> different tokenizer, label set, or normalization; mismatching them fails quietly with
> wrong output rather than loudly with an error.

### quantize

```python
model.quantize(mode=None, config=None, filters=None, **kwargs)
```

Quantize an already-built model in place. Passing `quantization=` to `from_weights` is
usually better, since it avoids materializing float weights first. See
[Quantization](quantization.md).

## Generation

### BaseGeneration

```python
model.generate(input_ids, attention_mask=None, max_new_tokens=None,
               eos_token_id=None, sampler=None, seed=None, **prefill_inputs)
```

Backend-agnostic autoregressive decoding for **decoder-only** models, the counterpart to
Hugging Face's `GenerationMixin`. Any extra tensors a multimodal model needs, pixel values
or audio features, ride along in `**prefill_inputs`, which is why a VLM call looks like
`model.generate(**inputs, max_new_tokens=64)`.

**Parameters**

- **input_ids**: the prompt token ids from a tokenizer or processor.
- **attention_mask** (*optional*): padding mask for batched prompts.
- **max_new_tokens** (`int`, *optional*): decode budget.
- **eos_token_id** (`int`, *optional*): stop token, defaulting to the model's own.
- **sampler** (*optional*): a sampler from `kerasformers.samplers`; greedy if omitted.
- **seed** (`int`, *optional*): seed for stochastic samplers.

### BaseSeq2SeqGeneration

```python
model.generate(encoder_inputs, decoder_input_ids, max_new_tokens=None,
               eos_token_id=None, sampler=None, seed=None)
model.encode(encoder_inputs)
```

The encoder-decoder flavor, used by [Whisper](whisper.md),
[Speech2Text](speech2text.md), and [Moonshine](moonshine.md). `encode` runs the encoder
once so you can decode repeatedly against the same audio.

Those speech models wrap this in a friendlier `generate(audio, processor, ...)` that owns
the whole pipeline; see their pages.

## Preprocessing

All preprocessors share `PreprocessorMixin`, so they also get `from_weights`,
`from_release`, and `from_hf`.

### BaseTokenizer

```python
tokenizer(inputs)
tokenizer.decode(ids, skip_special_tokens=True)
```

Text to token ids and back. Subclasses add `encode`, chat templating, and any
model-specific parsing (LocateAnything's `parse_boxes`, for example).

### BaseImageProcessor

```python
processor(images)
```

Images to `pixel_values`. Beyond the call, it carries the shared, backend-agnostic pixel
helpers every image model reuses: `resize`, `center_crop`, `pad`, `rescale`,
`normalize_image`, `rescale_and_normalize`, `preprocess_image`, and `stack_images` for
batching. Normalization constants live here too (`IMAGENET_STANDARD_MEAN`,
`OPENAI_CLIP_MEAN`, and friends).

Task-specific post-processing is added by subclasses:
`post_process_object_detection`, `post_process_semantic_segmentation`,
`post_process_depth_estimation`, `post_process_masks`.

### BaseAudioFeatureExtractor

```python
extractor(raw_speech, sampling_rate=16000)
```

Waveform to model input. What that means depends on the model: a log-mel spectrogram for
[Whisper](whisper.md) and [Granite Speech](granite_speech.md), normalized filterbanks for
[Speech2Text](speech2text.md), and the raw waveform itself for
[Moonshine](moonshine.md), which has no spectrogram step.

**`sampling_rate` tells the extractor what you are handing it; it does not resample.**
Feed 44.1 kHz audio while claiming 16 kHz and you get a confident, wrong transcript.

### BaseProcessor

```python
processor(text=None, images=None, audio=None, conversation=None, ...)
processor.decode(...)
```

The composite that bundles a tokenizer with an image processor or audio feature extractor
and renders chat templates. Multimodal models expose this as the single object you call.
Components are declared as class attributes, so `processor.tokenizer` and
`processor.image_processor` are always reachable.

## Quantization

### Quantizer

```python
quantizer.quantize(weight, axis=0)
```

Base for the weight-only quantizers. Helpers `normalize_axes(axis, ndim)` and
`single_axis(axis, ndim)` resolve contraction axes. The quantized layers built on this
(`QuantizedDense`, `QuantizedEinsumDense`, `QuantizedEmbedding`, `QuantizedExperts`) and
the `quantize_model` / `save_quantized` / `quantize_and_load` entry points are covered in
[Quantization](quantization.md).

## Attention

### fused_attention

```python
fused_attention(query, key, value, scale, attention_mask=None, soft_cap=None,
                dropout=None, training=None, attn_implementation=None)
```

Scaled dot-product attention with a selectable backend kernel, used by every attention
layer in the library so a single implementation choice applies everywhere.

- **scale** (`float`): the `1/sqrt(head_dim)` factor, applied inside.
- **attention_mask** (*optional*): additive mask broadcastable to `(B, heads, T_q, T_kv)`.
- **soft_cap** (`float`, *optional*): logit soft-capping, used by Gemma 2.
- **attn_implementation** (`str`, *optional*): pick the kernel; pass it through `from_weights` to set it model-wide.

See also [Utilities](utils.md) for the image, video, visualization, and label helpers.
