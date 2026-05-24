"""On-the-fly weight conversion for Qwen3.5 (HF safetensors -> Keras).

Qwen3.5 ships multimodal; the text tower lives under ``model.language_model.*``
(vision under ``model.visual.*`` and an ``mtp.*`` head — both ignored here).
Linear layers carry a Gated-DeltaNet block; every ``full_attention_interval``-th
layer carries gated full attention.
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


def transfer_qwen3_5_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        _build_model(keras_model)
    state = hf_state_dict
    pre = "model.language_model"

    keras_model.embed_tokens.embeddings.assign(_np(state, f"{pre}.embed_tokens.weight"))
    for i, layer in enumerate(keras_model.decoder_layers):
        p = f"{pre}.layers.{i}"
        _assign_rmsnorm(
            layer.input_layernorm, _np(state, f"{p}.input_layernorm.weight")
        )
        _assign_rmsnorm(
            layer.post_attention_layernorm,
            _np(state, f"{p}.post_attention_layernorm.weight"),
        )
        _assign_dense(layer.mlp.gate_proj, _np(state, f"{p}.mlp.gate_proj.weight"))
        _assign_dense(layer.mlp.up_proj, _np(state, f"{p}.mlp.up_proj.weight"))
        _assign_dense(layer.mlp.down_proj, _np(state, f"{p}.mlp.down_proj.weight"))

        if layer.layer_type == "full_attention":
            attn = layer.self_attn
            _assign_dense(attn.q_proj, _np(state, f"{p}.self_attn.q_proj.weight"))
            _assign_dense(attn.k_proj, _np(state, f"{p}.self_attn.k_proj.weight"))
            _assign_dense(attn.v_proj, _np(state, f"{p}.self_attn.v_proj.weight"))
            _assign_dense(attn.o_proj, _np(state, f"{p}.self_attn.o_proj.weight"))
            _assign_rmsnorm(attn.q_norm, _np(state, f"{p}.self_attn.q_norm.weight"))
            _assign_rmsnorm(attn.k_norm, _np(state, f"{p}.self_attn.k_norm.weight"))
        else:
            la = layer.linear_attn
            _assign_dense(
                la.in_proj_qkv, _np(state, f"{p}.linear_attn.in_proj_qkv.weight")
            )
            _assign_dense(la.in_proj_z, _np(state, f"{p}.linear_attn.in_proj_z.weight"))
            _assign_dense(la.in_proj_b, _np(state, f"{p}.linear_attn.in_proj_b.weight"))
            _assign_dense(la.in_proj_a, _np(state, f"{p}.linear_attn.in_proj_a.weight"))
            _assign_dense(la.out_proj, _np(state, f"{p}.linear_attn.out_proj.weight"))
            _assign_rmsnorm(la.norm, _np(state, f"{p}.linear_attn.norm.weight"))
            la.conv_weight.assign(
                _np(state, f"{p}.linear_attn.conv1d.weight").squeeze(1)
            )
            la.A_log.assign(_np(state, f"{p}.linear_attn.A_log"))
            la.dt_bias.assign(_np(state, f"{p}.linear_attn.dt_bias"))

    _assign_rmsnorm(keras_model.norm, _np(state, f"{pre}.norm.weight"))

    lm_head = getattr(keras_model, "lm_head", None)
    if lm_head is not None and "lm_head.weight" in state:
        _assign_dense(lm_head, _np(state, "lm_head.weight"))
