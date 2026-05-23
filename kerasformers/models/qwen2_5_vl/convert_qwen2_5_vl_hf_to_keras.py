"""On-the-fly weight conversion for Qwen2.5-VL (HF safetensors -> Keras).

Text decoder transfer is identical to Qwen2-VL; only the vision tower keys
differ: RMSNorm norms (weight only), a SwiGLU MLP (gate/up/down with bias),
and an RMSNorm ``merger.ln_q``.
"""

from kerasformers.models.qwen2_vl.convert_qwen2_vl_hf_to_keras import (
    _assign_dense,
    _assign_rmsnorm,
    _build_model,
    _normalize_state,
    _np,
    _transfer_text,
)


def transfer_qwen2_5_vl_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        _build_model(keras_model)
    state = _normalize_state(hf_state_dict)

    visual = keras_model.visual
    conv = _np(state, "visual.patch_embed.proj.weight")
    visual.patch_embed.proj.kernel.assign(conv.reshape(conv.shape[0], -1).T)
    for i, block in enumerate(visual.blocks):
        p = f"visual.blocks.{i}"
        _assign_rmsnorm(block.norm1, _np(state, f"{p}.norm1.weight"))
        _assign_rmsnorm(block.norm2, _np(state, f"{p}.norm2.weight"))
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
            block.mlp.gate_proj,
            _np(state, f"{p}.mlp.gate_proj.weight"),
            _np(state, f"{p}.mlp.gate_proj.bias"),
        )
        _assign_dense(
            block.mlp.up_proj,
            _np(state, f"{p}.mlp.up_proj.weight"),
            _np(state, f"{p}.mlp.up_proj.bias"),
        )
        _assign_dense(
            block.mlp.down_proj,
            _np(state, f"{p}.mlp.down_proj.weight"),
            _np(state, f"{p}.mlp.down_proj.bias"),
        )
    _assign_rmsnorm(visual.merger.ln_q, _np(state, "visual.merger.ln_q.weight"))
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
