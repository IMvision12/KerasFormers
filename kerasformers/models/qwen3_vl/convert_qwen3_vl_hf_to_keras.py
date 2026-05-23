"""On-the-fly weight conversion for Qwen3-VL (HF safetensors -> Keras)."""

from kerasformers.models.qwen2_vl.convert_qwen2_vl_hf_to_keras import (
    _assign_dense,
    _assign_layernorm,
    _assign_rmsnorm,
    _build_model,
    _normalize_state,
    _np,
)


def transfer_qwen3_vl_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        _build_model(keras_model)
    state = _normalize_state(hf_state_dict)

    # ---- Vision tower ----
    visual = keras_model.visual
    conv = _np(state, "visual.patch_embed.proj.weight")
    visual.patch_embed.proj.kernel.assign(conv.reshape(conv.shape[0], -1).T)
    visual.patch_embed.proj.bias.assign(_np(state, "visual.patch_embed.proj.bias"))
    visual.pos_embed.assign(_np(state, "visual.pos_embed.weight"))

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
            block.mlp.linear_fc1,
            _np(state, f"{p}.mlp.linear_fc1.weight"),
            _np(state, f"{p}.mlp.linear_fc1.bias"),
        )
        _assign_dense(
            block.mlp.linear_fc2,
            _np(state, f"{p}.mlp.linear_fc2.weight"),
            _np(state, f"{p}.mlp.linear_fc2.bias"),
        )

    def _merger(m, prefix):
        _assign_layernorm(
            m.norm,
            _np(state, f"{prefix}.norm.weight"),
            _np(state, f"{prefix}.norm.bias"),
        )
        _assign_dense(
            m.linear_fc1,
            _np(state, f"{prefix}.linear_fc1.weight"),
            _np(state, f"{prefix}.linear_fc1.bias"),
        )
        _assign_dense(
            m.linear_fc2,
            _np(state, f"{prefix}.linear_fc2.weight"),
            _np(state, f"{prefix}.linear_fc2.bias"),
        )

    _merger(visual.merger, "visual.merger")
    for j, dm in enumerate(visual.deepstack_mergers):
        _merger(dm, f"visual.deepstack_merger_list.{j}")

    # ---- Text decoder (Qwen3: q_norm/k_norm, no qkv bias) ----
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
        _assign_dense(attn.q_proj, _np(state, f"{p}.self_attn.q_proj.weight"))
        _assign_dense(attn.k_proj, _np(state, f"{p}.self_attn.k_proj.weight"))
        _assign_dense(attn.v_proj, _np(state, f"{p}.self_attn.v_proj.weight"))
        _assign_dense(attn.o_proj, _np(state, f"{p}.self_attn.o_proj.weight"))
        _assign_rmsnorm(attn.q_norm, _np(state, f"{p}.self_attn.q_norm.weight"))
        _assign_rmsnorm(attn.k_norm, _np(state, f"{p}.self_attn.k_norm.weight"))
        _assign_dense(layer.mlp.gate_proj, _np(state, f"{p}.mlp.gate_proj.weight"))
        _assign_dense(layer.mlp.up_proj, _np(state, f"{p}.mlp.up_proj.weight"))
        _assign_dense(layer.mlp.down_proj, _np(state, f"{p}.mlp.down_proj.weight"))
    _assign_rmsnorm(lm.norm, _np(state, "model.norm.weight"))

    lm_head = getattr(keras_model, "lm_head", None)
    if lm_head is not None and "lm_head.weight" in state:
        _assign_dense(lm_head, _np(state, "lm_head.weight"))
