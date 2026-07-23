---
hide:
  - navigation
---

# Loading Weights

Every model and preprocessor loads through one call:

```python
model = SegFormerSemanticSegment.from_weights("segformer_b0_ade_512")
```

Behind that one call sit **three ways** weights actually reach the model. You rarely choose
between them explicitly, but knowing which one a model uses explains what you see on first
load, and how long it takes.

| # | Way | What happens | Used by |
|---|---|---|---|
| 1 | [**On the fly**](#1-on-the-fly-conversion) | A release variant whose weights live on the Hub. The original safetensors are downloaded and converted in process, every load. | The LLMs and VLMs: Qwen, Llama, Gemma, DeepSeek, GLM, Mistral, ... |
| 2 | [**`hf:` prefix**](#2-the-hf-prefix) | Any Hub repo, named explicitly. Same conversion machinery, but the repo is yours to pick. | Fine-tunes and community weights, for any architecture |
| 3 | [**Release**](#3-release-weights) | A pre-converted `.weights.h5` downloaded from the project's GitHub Releases. No conversion at load time. | The vision, detection, segmentation, and speech models |

Ways 1 and 3 are both reached by a plain variant id; which one you get is a property of the
model, set per variant in its `BASE_WEIGHT_CONFIG`. Way 2 is the one you opt into, by
prefixing `hf:`.

```python
model = Qwen3Generate.from_weights("qwen3-4b")                                # 1: on the fly
model = SegFormerSemanticSegment.from_weights("hf:<user>/my-finetune")        # 2: hf:
model = SegFormerSemanticSegment.from_weights("segformer_b0_ade_512")         # 3: release
```

## 1. On-the-fly conversion

**Used by every LLM and VLM in the library.** These variants carry an `hf_id` rather than a
URL, so `from_weights` downloads the original safetensors from the Hub and converts them in
memory:

```python
QWEN2_WEIGHTS_URLS["qwen2-0.5b"]
# {'hf_id': 'Qwen/Qwen2-0.5B', 'gated': False, 'safetensors': True}
```

```python
model = Qwen3Generate.from_weights("qwen3-4b")
```

The reason is **size**. A pre-converted `.weights.h5` for a 4B model is ~8 GB and a 120B one
is hundreds; GitHub Releases is not a sensible place to mirror that, and every user would
pay to download a second copy of weights the Hub already serves. Converting on arrival costs
CPU time instead of storage, and the Hub download is cached like any other.

The tradeoff is that **the conversion runs on every load**. To pay it once:

```python
model = Qwen3Generate.from_weights("qwen3-4b", cache_converted=True)
```

That stores the converted result under `$KERASFORMERS_HOME/converted` and rebuilds from it
next time, skipping both the download and the conversion. See [Caching](#caching).

Some families are **gated**. The variant is marked in its config, and you need to accept the
license on the Hub and authenticate:

```python
GEMMA_WEIGHTS_URLS["gemma-2b"]
# {'hf_id': 'google/gemma-2b', 'gated': True, 'safetensors': True}
```

```shell
huggingface-cli login          # or: export HF_TOKEN=...
```

Every model page whose weights load this way carries a red banner saying so.

## 2. The `hf:` prefix

Way 1 picks the repo for you. Prefix any Hub repo with `hf:` and you pick it yourself; it is
fetched, converted, and loaded in the same call, through the same machinery. There is no
offline conversion step and no intermediate file to manage.

```python
model = Qwen2Generate.from_weights("hf:Qwen/Qwen2-1.5B-Instruct")
tokenizer = Qwen2Tokenizer.from_weights("hf:Qwen/Qwen2-1.5B-Instruct")
```

What that does:

1. Downloads `config.json` and checks its `model_type` against the class.
2. Maps that config into constructor arguments (`config_from_hf`) and builds the model.
3. Downloads the safetensors and assigns every tensor into the Keras layers (`transfer_from_hf`), transposing, splitting, and fusing as each architecture requires.

This runs through `huggingface_hub` only. **`transformers` and `torch` are never imported**,
so the conversion happens wherever you are running, on any backend.

Because step 2 reads the repo's own config, a fine-tune with a different class count,
vocabulary, or image size needs no extra arguments: it is read off the checkpoint. That is
what makes community weights work.

```python
model = SegFormerSemanticSegment.from_weights("hf:<user>/segformer-my-dataset")
```

Point a class at the wrong checkpoint and it fails immediately rather than deep inside the
transfer:

```python
SegFormerSemanticSegment.from_weights("hf:openai/clip-vit-base-patch16")
```

```
ValueError: SegFormerSemanticSegment can only load HF models whose config.json model_type
is segformer, but 'openai/clip-vit-base-patch16' has model_type='clip'. This kerasformers
class is the wrong destination for that checkpoint.
```

Classification backbones come from **timm-style** repos instead, which carry no
`model_type`. There the variant is inferred from the repo name, and `variant=` overrides it
when a fine-tune does not follow the timm naming convention:

```python
model = ResNetImageClassify.from_weights("hf:timm/resnet50.a1_in1k")
model = ResNetImageClassify.from_weights("hf:<user>/my-resnet", variant="resnet50")
```

> **Load the processor from the same source as the model.** A fine-tune can ship a
> different tokenizer, label set, or normalization; mismatching them fails quietly with
> wrong output rather than loudly with an error.

## 3. Release weights

**Used by the vision, detection, segmentation, and speech models.** These variants carry a
URL to a `.weights.h5` already converted to Keras format and published on the project's
GitHub Releases:

```python
SEGFORMER_WEIGHTS_URLS["segformer_b0_cityscapes_1024"]
# {'url': 'https://github.com/IMvision12/KerasFormers/releases/download/segformer/...h5'}
```

```python
model = SegFormerSemanticSegment.from_weights("segformer_b0_ade_512")
```

Nothing is converted at load time: the file is downloaded once, cached in `~/.downloads`,
and handed to Keras's own `load_weights`. That makes this the fastest of the three, which is
why it is the default wherever the checkpoint is small enough to mirror.

Sharded releases work the same way: a `.weights.json` index pulls every shard it lists from
the same release before loading.

## Release variants

Ways 1 and 3 both take a **release variant**: a short id listed in the Model Variants table
on each model page. It resolves against two dicts on the class, `BASE_MODEL_CONFIG` for the
constructor arguments and `BASE_WEIGHT_CONFIG` for where the weights come from.

The variant carries the architecture with it, so you do not pass `hidden_size`,
`num_layers`, or `num_classes`. Get the id wrong and the error lists every id that exists:

```python
SegFormerSemanticSegment.from_weights("segformer_b0_cityscapes")
```

```
ValueError: Unknown variant 'segformer_b0_cityscapes' for SegFormerSemanticSegment.
Available variants: ['segformer_b0_ade_512', 'segformer_b0_cityscapes_1024',
'segformer_b0_cityscapes_768', 'segformer_b1_ade_512', ...]
```

`BASE_WEIGHT_CONFIG` is also where you can check which way any given variant uses: a `url`
key means way 3, an `hf_id` key means way 1.

## Caching

Downloads are cached, so the second load of the same weights is local:

- Release `.h5` and `.json` files land in `~/.downloads`.
- Hub files use the standard `huggingface_hub` cache, so they are shared with anything else on the machine that has pulled the same repo.

Conversion itself is **not** cached by default, which is what makes way 1 slower on the
second load than way 3. For a checkpoint you load repeatedly, `cache_converted=True` stores
the converted result under `$KERASFORMERS_HOME/converted` (default
`~/.cache/kerasformers/converted`) and rebuilds from it next time, skipping both download and
conversion:

```python
model = Qwen3Generate.from_weights("qwen3-4b", cache_converted=True)          # way 1
model = Qwen3Generate.from_weights("hf:Qwen/Qwen3-8B", cache_converted=True)  # way 2
```

It works for both, since either can involve a conversion. Way 3 has nothing to cache beyond
the downloaded file.

The cache key includes the source identity, the backend and dtype, and the quantization
recipe, so it cannot hand back a stale or differently configured model. For an `hf:` id the
source identity is the resolved **commit SHA**, so a repo that moves invalidates the entry.
A miss falls back to the normal path silently. On an ephemeral machine (Colab, CI) point
`KERASFORMERS_HOME` at persistent storage or the cache buys you nothing.

## Loading big checkpoints

Four independent flags, composable, for checkpoints that do not comfortably fit:

| Flag | Trades against | Effect |
|---|---|---|
| `load_dtype="bfloat16"` | Device memory | Builds under a bf16 policy so a bf16 checkpoint stays ~2 bytes/param instead of being upcast to fp32. |
| `quantization="int8"` | Device memory | Weight-only quantization of Dense and Embedding layers, roughly 4x, or 8x for `"int4"`. See [Quantization](quantization.md). |
| `low_memory=True` | Peak RAM | With `quantization`, streams weights straight into int storage so the full float model is never built. |
| `low_disk=True` | Local disk | Downloads a sharded checkpoint one shard at a time, converting and evicting each before the next, so peak disk is about one shard. |

```python
model = Qwen3Generate.from_weights(
    "hf:Qwen/Qwen3-8B",
    load_dtype="bfloat16",
    quantization="int8",
    low_memory=True,
    low_disk=True,
)
```

These are memory and disk optimizations, not speed ones. Weight-only quantization in
particular dequantizes on the fly, so it buys footprint, not throughput.

## Architecture only, and partial loads

`load_weights=False` builds the architecture with random initialization. For an `hf:` id
the config is still fetched to size the model, but the weight files are not downloaded.

```python
model = SegFormerSemanticSegment.from_weights("segformer_b0_ade_512", load_weights=False)
```

`skip_mismatch=True` loads everything whose shape agrees and leaves the rest at its
initializer, which is how you keep a pretrained backbone while swapping the head:

```python
model = SegFormerSemanticSegment.from_weights(
    "segformer_b0_ade_512", num_classes=7, skip_mismatch=True
)
print(model.output_shape)
```

```
List of objects that could not be loaded:
[<Conv2D name=head_classifier, built=True>]
(None, 512, 512, 7)
```

Either way it tells you what it skipped. The pre-converted `.h5` path reports it as the
Keras warning above; the converter path (`hf:` ids and the safetensors releases) prints its
own line instead:

```
[from_weights] skip_mismatch: left 2 weight(s) at their initialized values due to
shape mismatch (e.g. a resized head): [...]
```

Read that list. It is your only signal that the head really was the only thing left
untrained.

## Being explicit

`from_weights` dispatches to two classmethods you can also call directly, when you would
rather the source be visible at the call site than encoded in a prefix:

```python
model = SegFormerSemanticSegment.from_release("segformer_b0_ade_512")
model = SegFormerSemanticSegment.from_hf("nvidia/segformer-b0-finetuned-ade-512-512")
```

Full signatures are in [Main Classes](main_classes.md#loading-weights).
