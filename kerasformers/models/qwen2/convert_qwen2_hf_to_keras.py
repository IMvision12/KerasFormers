"""On-the-fly weight conversion for Qwen2 (HF safetensors -> Keras).

Follows the library's name-mapped convention (see CLIP / DINOv3 / DETR): driven
off the Keras model's own weights, each weight's hierarchical ``path`` is mapped
to the HF tensor name and assigned via the shared ``transfer_weights`` helper.

kerasformers uses its own layer names (``attention.query`` etc.), so
``hf_weight_name`` bridges them to HF's (``self_attn.q_proj`` etc.).
"""

import numpy as np

from kerasformers.weight_utils.custom_exception import WeightMappingError
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights

# kerasformers layer-name segment -> HF segment (applied after "/"->".").
_LAYER_MAP = {
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
}


def hf_weight_name(path):
    """Map a Keras weight ``path`` to its HuggingFace tensor name."""
    rest = path.split("/", 1)[1]  # drop the model-name root
    if rest.startswith("token_embedding"):
        return "model.embed_tokens.weight"
    if rest.startswith("final_norm"):
        return "model.norm.weight"
    if rest.startswith("lm_head"):
        return "lm_head.weight"
    rest = rest.replace("decoder_layer_", "layers.").replace("/", ".")
    for old, new in _LAYER_MAP.items():
        rest = rest.replace(old, new)
    return "model." + rest.replace(".kernel", ".weight")


def build_model(model):
    """Materialize weights with a tiny dummy forward."""
    model({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})


def transfer_qwen2_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        build_model(keras_model)
    for weight in keras_model.weights:
        name = hf_weight_name(weight.path)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        transfer_weights(weight.path, weight, hf_state_dict[name])


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
    del hf
    gc.collect()

    model = Qwen2Generate.from_weights("hf:" + HF_ID)
    k_logits = ops.convert_to_numpy(model({"input_ids": ids})["logits"])
    diff = float(np.max(np.abs(hf_logits - k_logits)))
    print(f"max abs logit diff: {diff:.6e}")
    assert diff < 1e-2
