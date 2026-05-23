"""On-the-fly weight conversion for Qwen2 (HF safetensors -> Keras).

``transfer_qwen2_weights`` is what ``Qwen2Model.transfer_from_hf`` calls at load
time. Raw checkpoint keys: ``model.embed_tokens``, ``model.layers.{i}.*``,
``model.norm``, and ``lm_head.weight`` (absent when the head is tied).
"""

import numpy as np


def _np(state, key):
    v = state[key]
    if not isinstance(v, np.ndarray):
        v = v.detach().cpu().numpy()
    return v.astype("float32")


def _assign_dense(dense, weight, bias=None):
    dense.kernel.assign(np.asarray(weight).T)  # torch (out,in) -> keras (in,out)
    if bias is not None:
        dense.bias.assign(np.asarray(bias))


def _assign_rmsnorm(norm, weight):
    norm.weight.assign(np.asarray(weight))


def _build_model(model):
    """Materialize weights with a tiny dummy forward."""
    model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})


def transfer_qwen2_weights(keras_model, hf_state_dict):
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
    _assign_rmsnorm(keras_model.norm, _np(state, "model.norm.weight"))

    lm_head = getattr(keras_model, "lm_head", None)
    if lm_head is not None and "lm_head.weight" in state:
        _assign_dense(lm_head, _np(state, "lm_head.weight"))


if __name__ == "__main__":
    import gc

    import torch
    from keras import ops
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from .qwen2_model import Qwen2Generate

    HF_ID = "Qwen/Qwen2-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(HF_ID)
    text = tok.apply_chat_template(
        [{"role": "user", "content": "Hello!"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="np")["input_ids"].astype("int64")

    hf = Qwen2ForCausalLM.from_pretrained(HF_ID, torch_dtype=torch.float32).eval()
    hf.set_attn_implementation("eager")
    with torch.no_grad():
        hf_logits = hf(torch.tensor(ids)).logits.float().numpy()
    state = {k: v.detach().cpu().numpy() for k, v in hf.state_dict().items()}
    del hf
    gc.collect()

    model = Qwen2Generate.from_weights("hf:" + HF_ID)
    k_logits = ops.convert_to_numpy(model({"input_ids": ids})["logits"])
    diff = float(np.max(np.abs(hf_logits - k_logits)))
    print(f"max abs logit diff: {diff:.6e}")
    assert diff < 1e-2
