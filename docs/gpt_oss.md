# GPT-OSS (mixture-of-experts LLM)

OpenAI's GPT-OSS in **pure Keras 3**: a mixture-of-experts decoder-only language
model with grouped-query attention, learned per-head **attention sinks**,
alternating sliding-window / full causal attention, **YaRN** rotary positions,
and top-k sparse MoE feed-forwards. One implementation runs unmodified on
**TensorFlow / Torch / JAX**. Weights are **converted on the fly** from the
Hugging Face repos (including their MXFP4 quantization): nothing is
re-hosted.

**Paper / model card**: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

| Class | Module | Output |
|---|---|---|
| `GptOssModel` | `kerasformers.models.gpt_oss` | `{"last_hidden_state": (B, L, embed_dim)}` |
| `GptOssGenerate` | `kerasformers.models.gpt_oss` | `{"logits": (B, L, vocab), "last_hidden_state": ...}` + `.generate()` |
| `GptOssTokenizer` | `kerasformers.models.gpt_oss` | `o200k_harmony` → `input_ids` / `attention_mask` |

`GptOssModel` is a subclassed (imperative) `SubclassedBaseModel` whose forward
runs eagerly with `keras.ops`; `GptOssGenerate` adds an (untied) LM head and
greedy `.generate()` with a KV cache that respects each layer's sliding window.

## Architecture notes

- **MoE feed-forward**: a top-`num_experts_per_tok` (4) router selects experts
  whose softmax weights combine per-expert outputs. The expert activation is
  GPT-OSS's clamped gated-SiLU on the interleaved gate/up halves
  (`(up+1) * gate*sigmoid(1.702*gate)`, clamp limit 7). Experts are evaluated
  densely (every expert, masked by the routing weights): exact, backend-agnostic,
  and fine for short prompts; long sequences over all 32/128 experts are heavy.
- **Attention sinks**: a learned per-head logit is appended to the attention
  scores before softmax and dropped afterward, letting a head attend to "nothing".
- **Alternating attention**: even layers use a sliding window of `sliding_window`
  (128) tokens; odd layers use full causal attention.
- **YaRN rotary** scaling (factor 32, β_fast 32, β_slow 1, original context 4096)
  with the mscale cos/sin factor.
- **MXFP4**: the official repos store the experts in MXFP4 (4-bit blocks + e8m0
  scales). They are dequantized to float32 on load (bit-exact port of HF's
  `convert_moe_packed_tensors`), so Keras runs the experts in full precision.

## Loading (on the fly, no release weights)

```python
from kerasformers.models.gpt_oss import GptOssGenerate, GptOssTokenizer

# Downloads + converts openai/gpt-oss-20b once, then caches under
# ~/.cache/kerasformers/. (Accept the model license + set HF_TOKEN if needed.)
model = GptOssGenerate.from_weights("gpt-oss-20b")        # or "hf:openai/gpt-oss-20b"
tok = GptOssTokenizer()

inputs = tok([{"role": "user", "content": "Hello!"}])
ids = model.generate(inputs["input_ids"], max_new_tokens=64)
print(tok.decode(ids[0]))
```

### Available variants

| Variant | layers | experts | top-k | heads (Q/KV) |
|---|---|---|---|---|
| `gpt-oss-20b` | 24 | 32 | 4 | 64 / 8 |
| `gpt-oss-120b` | 36 | 128 | 4 | 64 / 8 |

## Verified parity

The 20B/120B checkpoints are too large to run on a dev box, so the architecture
is validated against a **tiny random `GptOssForCausalLM`** (Hugging Face,
eager attention) exercising the MoE router, attention sinks, the sliding window
(sequence longer than the window), YaRN, and GQA:

| Check | Result |
|---|---|
| `GptOssGenerate` logits vs HF (tiny random) | max \|Δ\| **1.8e-7**, argmax 100% agree |
| MXFP4 dequant vs HF `convert_moe_packed_tensors` | max \|Δ\| **0.0** |
| Build + forward + `.generate()` on TF / Torch / JAX | ✓ |
