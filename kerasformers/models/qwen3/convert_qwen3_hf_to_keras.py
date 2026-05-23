"""On-the-fly weight conversion for Qwen3 (HF safetensors -> Keras).

Same key layout as Qwen2 but the attention has no q/k/v bias and adds
``self_attn.q_norm`` / ``self_attn.k_norm``.
"""

import numpy as np


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


def _build_model(model):
    model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})


def transfer_qwen3_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        _build_model(keras_model)
    state = hf_state_dict

    keras_model.embed_tokens.embeddings.assign(_np(state, "model.embed_tokens.weight"))
    for i, layer in enumerate(keras_model.decoder_layers):
        p = f"model.layers.{i}"
        _assign_rmsnorm(
            layer.input_layernorm, _np(state, f"{p}.input_layernorm.weight")
        )
        _assign_rmsnorm(
            layer.post_attention_layernorm,
            _np(state, f"{p}.post_attention_layernorm.weight"),
        )
        attn = layer.self_attn
        _assign_dense(attn.q_proj, _np(state, f"{p}.self_attn.q_proj.weight"))
        _assign_dense(attn.k_proj, _np(state, f"{p}.self_attn.k_proj.weight"))
        _assign_dense(attn.v_proj, _np(state, f"{p}.self_attn.v_proj.weight"))
        _assign_dense(attn.o_proj, _np(state, f"{p}.self_attn.o_proj.weight"))
        _assign_rmsnorm(attn.q_norm, _np(state, f"{p}.self_attn.q_norm.weight"))
        _assign_rmsnorm(attn.k_norm, _np(state, f"{p}.self_attn.k_norm.weight"))
        _assign_dense(layer.mlp.gate_proj, _np(state, f"{p}.mlp.gate_proj.weight"))
        _assign_dense(layer.mlp.up_proj, _np(state, f"{p}.mlp.up_proj.weight"))
        _assign_dense(layer.mlp.down_proj, _np(state, f"{p}.mlp.down_proj.weight"))
    _assign_rmsnorm(keras_model.norm, _np(state, "model.norm.weight"))

    lm_head = getattr(keras_model, "lm_head", None)
    if lm_head is not None and "lm_head.weight" in state:
        _assign_dense(lm_head, _np(state, "lm_head.weight"))
