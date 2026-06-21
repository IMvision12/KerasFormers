import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights


def hf_key_for(path):
    """Map a keras weight path (dot-joined, model prefix stripped) to its HF
    state-dict key. MoonViT -> ``vision_model.*``, projector -> ``mlp1.{0,1,3}``,
    Qwen2 LLM -> ``language_model.model.*``."""
    p = path
    if p.startswith("vision_model."):
        p = p.replace(
            "vision_model.patch_proj.kernel", "vision_model.patch_embed.proj.weight"
        )
        p = p.replace(
            "vision_model.patch_proj.bias", "vision_model.patch_embed.proj.bias"
        )
        p = p.replace("vision_model.pos_emb", "vision_model.patch_embed.pos_emb.weight")
        p = p.replace(
            "vision_model.final_norm.gamma",
            "vision_model.encoder.final_layernorm.weight",
        )
        p = p.replace(
            "vision_model.final_norm.beta", "vision_model.encoder.final_layernorm.bias"
        )
        p = p.replace("vision_model.block_", "vision_model.encoder.blocks.")
        p = p.replace(".norm0.gamma", ".norm0.weight").replace(
            ".norm0.beta", ".norm0.bias"
        )
        p = p.replace(".norm1.gamma", ".norm1.weight").replace(
            ".norm1.beta", ".norm1.bias"
        )
        p = p.replace(".wqkv.kernel", ".wqkv.weight").replace(
            ".wo.kernel", ".wo.weight"
        )
        p = p.replace(".mlp.fc0.kernel", ".mlp.fc0.weight").replace(
            ".mlp.fc1.kernel", ".mlp.fc1.weight"
        )
        return p
    if p.startswith("mlp1"):
        p = p.replace("mlp1_norm.gamma", "mlp1.0.weight").replace(
            "mlp1_norm.beta", "mlp1.0.bias"
        )
        p = p.replace("mlp1_fc1.kernel", "mlp1.1.weight").replace(
            "mlp1_fc1.bias", "mlp1.1.bias"
        )
        p = p.replace("mlp1_fc2.kernel", "mlp1.3.weight").replace(
            "mlp1_fc2.bias", "mlp1.3.bias"
        )
        return p
    p = p.replace(
        "token_embedding.embeddings", "language_model.model.embed_tokens.weight"
    )
    p = p.replace("final_norm.weight", "language_model.model.norm.weight")
    p = p.replace("decoder_layer_", "language_model.model.layers.")
    p = p.replace("attention.query", "self_attn.q_proj").replace(
        "attention.key", "self_attn.k_proj"
    )
    p = p.replace("attention.value", "self_attn.v_proj").replace(
        "attention.output_proj", "self_attn.o_proj"
    )
    p = p.replace("attention_norm", "input_layernorm").replace(
        "mlp_norm", "post_attention_layernorm"
    )
    p = (
        p.replace("mlp.gate", "mlp.gate_proj")
        .replace("mlp.up", "mlp.up_proj")
        .replace("mlp.down", "mlp.down_proj")
    )
    p = p.replace("kernel", "weight")
    return p


def build_for_transfer(keras_model):
    grid = np.array([[2, 2]], dtype="int64")
    pixel_values = np.zeros((4, 3, 14, 14), dtype="float32")
    img = keras_model.image_token_index
    input_ids = np.array([[img, 0, 0, 0]], dtype="int64")
    keras_model(
        {"input_ids": input_ids, "pixel_values": pixel_values, "image_grid_hws": grid}
    )


def transfer_locateanything_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        build_for_transfer(keras_model)
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        path = weight.path.split("/", 1)[1].replace("/", ".")
        key = hf_key_for(path)
        if key not in hf_state_dict:
            raise WeightMappingError(weight.path, key)
        value = hf_state_dict[key]
        if path.endswith("patch_proj.kernel"):
            weight.assign(np.transpose(np.asarray(value), (2, 3, 1, 0)))
        elif path.endswith("pos_emb"):
            weight.assign(np.asarray(value))
        else:
            transfer_weights(weight.path, weight, value)
