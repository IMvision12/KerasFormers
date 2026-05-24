"""On-the-fly weight conversion for Qwen2-VL (HF safetensors -> Keras).

``transfer_qwen2_vl_weights`` is what ``Qwen2VLModel.transfer_from_hf`` calls at
load time — there is no kerasformers release upload for this family. The raw
checkpoint uses the flat legacy key layout: text under ``model.*``, vision under
``visual.*``, and (for the 2B) a tied LM head (no ``lm_head.weight``).

The ``__main__`` block runs a local logit-parity check against the HF reference
(``transformers``) and is for verification only.
"""

import numpy as np

from .qwen2_vl_model import Qwen2VLModel


def _normalize_state(state):
    """Map keys to the legacy layout (``visual.*`` / ``model.*`` / ``lm_head.*``).

    The raw hub safetensors already use the legacy layout; an in-memory
    ``state_dict()`` from recent ``transformers`` uses ``model.visual.*`` /
    ``model.language_model.*``. Normalizing here makes the converter accept
    either, so runtime loading and local parity tests share one code path.
    """
    out = {}
    for k, v in state.items():
        if k.startswith("model.visual."):
            k = k[len("model.") :]
        elif k.startswith("model.language_model."):
            k = "model." + k[len("model.language_model.") :]
        out[k] = v
    return out


def _np(state, key):
    v = state[key]
    if not isinstance(v, np.ndarray):
        v = v.detach().cpu().numpy()
    return v.astype("float32")


def _assign_dense(dense, weight, bias=None):
    dense.kernel.assign(np.asarray(weight).T)
    if bias is not None:
        dense.bias.assign(np.asarray(bias))


def _assign_rmsnorm(norm, weight):
    norm.weight.assign(np.asarray(weight))


def _assign_layernorm(norm, weight, bias):
    norm.gamma.assign(np.asarray(weight))
    norm.beta.assign(np.asarray(bias))


def _build_model(model):
    """Materialize all weights with a minimal, self-consistent dummy forward."""
    m = model.spatial_merge_size
    h = w = 2 * m
    grid = np.array([[1, h, w]], dtype=np.int64)
    n_patches = h * w
    n_merged = n_patches // (m * m)
    pixel_values = np.zeros((n_patches, model.patch_dim), dtype="float32")
    input_ids = np.array([[0] + [model.image_token_id] * n_merged + [1]], dtype="int64")
    model(
        {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": grid,
        }
    )


def transfer_qwen2_vl_weights(keras_model, hf_state_dict):
    """Assign HF Qwen2-VL weights into ``keras_model`` (built if needed)."""
    if not keras_model.built or not keras_model.weights:
        _build_model(keras_model)

    state = _normalize_state(hf_state_dict)

    visual = keras_model.visual
    conv = _np(state, "visual.patch_embed.proj.weight")
    visual.patch_embed.proj.kernel.assign(conv.reshape(conv.shape[0], -1).T)
    for i, block in enumerate(visual.blocks):
        p = f"visual.blocks.{i}"
        _assign_layernorm(
            block.norm1, _np(state, f"{p}.norm1.weight"), _np(state, f"{p}.norm1.bias")
        )
        _assign_layernorm(
            block.norm2, _np(state, f"{p}.norm2.weight"), _np(state, f"{p}.norm2.bias")
        )
        _assign_dense(
            block.attn.qkv,
            _np(state, f"{p}.attn.qkv.weight"),
            _np(state, f"{p}.attn.qkv.bias"),
        )
        _assign_dense(
            block.attn.proj,
            _np(state, f"{p}.attn.proj.weight"),
            _np(state, f"{p}.attn.proj.bias"),
        )
        _assign_dense(
            block.mlp.fc1,
            _np(state, f"{p}.mlp.fc1.weight"),
            _np(state, f"{p}.mlp.fc1.bias"),
        )
        _assign_dense(
            block.mlp.fc2,
            _np(state, f"{p}.mlp.fc2.weight"),
            _np(state, f"{p}.mlp.fc2.bias"),
        )
    _assign_layernorm(
        visual.merger.ln_q,
        _np(state, "visual.merger.ln_q.weight"),
        _np(state, "visual.merger.ln_q.bias"),
    )
    _assign_dense(
        visual.merger.mlp_fc1,
        _np(state, "visual.merger.mlp.0.weight"),
        _np(state, "visual.merger.mlp.0.bias"),
    )
    _assign_dense(
        visual.merger.mlp_fc2,
        _np(state, "visual.merger.mlp.2.weight"),
        _np(state, "visual.merger.mlp.2.bias"),
    )

    _transfer_text(keras_model, state)


def _transfer_text(keras_model, state):
    """Transfer the Qwen2 text decoder + (untied) LM head. Shared with 2.5-VL."""
    lm = keras_model.language_model
    lm.embed_tokens.embeddings.assign(_np(state, "model.embed_tokens.weight"))
    for i, layer in enumerate(lm.decoder_layers):
        p = f"model.layers.{i}"
        _assign_rmsnorm(
            layer.input_layernorm, _np(state, f"{p}.input_layernorm.weight")
        )
        _assign_rmsnorm(
            layer.post_attention_layernorm,
            _np(state, f"{p}.post_attention_layernorm.weight"),
        )
        attn = layer.self_attn
        _assign_dense(
            attn.q_proj,
            _np(state, f"{p}.self_attn.q_proj.weight"),
            _np(state, f"{p}.self_attn.q_proj.bias"),
        )
        _assign_dense(
            attn.k_proj,
            _np(state, f"{p}.self_attn.k_proj.weight"),
            _np(state, f"{p}.self_attn.k_proj.bias"),
        )
        _assign_dense(
            attn.v_proj,
            _np(state, f"{p}.self_attn.v_proj.weight"),
            _np(state, f"{p}.self_attn.v_proj.bias"),
        )
        _assign_dense(attn.o_proj, _np(state, f"{p}.self_attn.o_proj.weight"))
        _assign_dense(layer.mlp.gate_proj, _np(state, f"{p}.mlp.gate_proj.weight"))
        _assign_dense(layer.mlp.up_proj, _np(state, f"{p}.mlp.up_proj.weight"))
        _assign_dense(layer.mlp.down_proj, _np(state, f"{p}.mlp.down_proj.weight"))
    _assign_rmsnorm(lm.norm, _np(state, "model.norm.weight"))

    lm_head = getattr(keras_model, "lm_head", None)
    if lm_head is not None and "lm_head.weight" in state:
        _assign_dense(lm_head, _np(state, "lm_head.weight"))


if __name__ == "__main__":
    import gc

    import torch
    from keras import ops
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    HF_ID = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"[1/4] Loading HF {HF_ID} (float32, cpu)")
    hf = Qwen2VLForConditionalGeneration.from_pretrained(
        HF_ID, torch_dtype=torch.float32
    ).eval()
    processor = AutoProcessor.from_pretrained(HF_ID)

    print("[2/4] Building Keras model + transferring weights")
    state = {k: v.detach().cpu().numpy() for k, v in hf.state_dict().items()}
    model = Qwen2VLModel.from_weights(
        HF_ID.replace("Qwen/", "").lower(), load_weights=False
    )
    transfer_qwen2_vl_weights(model, state)

    print("[3/4] Building a real image+text input")
    from PIL import Image

    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[img], return_tensors="pt")

    print("[4/4] Comparing logits")
    with torch.no_grad():
        hf_logits = hf(**inputs).logits.float().cpu().numpy()
    k_logits = ops.convert_to_numpy(
        model(
            {
                "input_ids": inputs["input_ids"].cpu().numpy(),
                "pixel_values": inputs["pixel_values"].float().cpu().numpy(),
                "image_grid_thw": inputs["image_grid_thw"].cpu().numpy(),
            }
        )["logits"]
    )
    diff = float(np.max(np.abs(hf_logits - k_logits)))
    print(f"  max abs logit diff: {diff:.6e}")
    assert diff < 1e-2, f"parity too high: {diff:.6e}"
    del hf, state
    gc.collect()
