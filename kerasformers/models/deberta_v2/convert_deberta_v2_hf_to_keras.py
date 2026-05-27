import re
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from kerasformers.weight_utils.custom_exception import WeightMappingError
from kerasformers.weight_utils.weight_transfer_torch_to_keras import transfer_weights

WEIGHT_NAME_MAPPING = {
    "embeddings/word_embeddings/embeddings": "embeddings.word_embeddings.weight",
    "embeddings/LayerNorm/gamma": "embeddings.LayerNorm.weight",
    "embeddings/LayerNorm/beta": "embeddings.LayerNorm.bias",
    "rel_embeddings/embeddings": "encoder.rel_embeddings.weight",
    "rel_embeddings_layernorm/gamma": "encoder.LayerNorm.weight",
    "rel_embeddings_layernorm/beta": "encoder.LayerNorm.bias",
    "conv/conv/kernel": "encoder.conv.conv.weight",
    "conv/conv/bias": "encoder.conv.conv.bias",
    "conv/LayerNorm/gamma": "encoder.conv.LayerNorm.weight",
    "conv/LayerNorm/beta": "encoder.conv.LayerNorm.bias",
    # masked-LM head (legacy cls.predictions.*); decoder is present + tied to embeddings
    "lm_head_dense/kernel": "cls.predictions.transform.dense.weight",
    "lm_head_dense/bias": "cls.predictions.transform.dense.bias",
    "lm_head_layernorm/gamma": "cls.predictions.transform.LayerNorm.weight",
    "lm_head_layernorm/beta": "cls.predictions.transform.LayerNorm.bias",
    "lm_head_decoder/kernel": "cls.predictions.decoder.weight",
    "lm_head_decoder/bias": "cls.predictions.bias",
    # context pooler + task heads
    "pooler_dense/kernel": "pooler.dense.weight",
    "pooler_dense/bias": "pooler.dense.bias",
    "classifier/kernel": "classifier.weight",
    "classifier/bias": "classifier.bias",
    "qa_outputs/kernel": "qa_outputs.weight",
    "qa_outputs/bias": "qa_outputs.bias",
}

_OPTIONAL_WEIGHTS = ("classifier", "qa_outputs", "lm_head", "pooler_dense")

_LAYER_MAP = {
    "attention_output_dense": "attention.output.dense",
    "intermediate_dense": "intermediate.dense",
    "output_dense": "output.dense",
    "attention_output_layernorm": "attention.output.LayerNorm",
    "output_layernorm": "output.LayerNorm",
}


def hf_name_for(path: str) -> Optional[str]:
    if path in WEIGHT_NAME_MAPPING:
        return WEIGHT_NAME_MAPPING[path]

    m = re.match(
        r"blocks_(\d+)_attention_self/blocks_\d+_(query_proj|key_proj|value_proj)/(kernel|bias)$",
        path,
    )
    if m:
        idx, proj, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"encoder.layer.{idx}.attention.self.{proj}.{suffix}"

    m = re.match(
        r"blocks_(\d+)_(attention_output_dense|intermediate_dense|output_dense)/(kernel|bias)$",
        path,
    )
    if m:
        idx, layer, w = m.groups()
        suffix = "weight" if w == "kernel" else "bias"
        return f"encoder.layer.{idx}.{_LAYER_MAP[layer]}.{suffix}"

    m = re.match(
        r"blocks_(\d+)_(attention_output_layernorm|output_layernorm)/(gamma|beta)$",
        path,
    )
    if m:
        idx, layer, w = m.groups()
        suffix = "weight" if w == "gamma" else "bias"
        return f"encoder.layer.{idx}.{_LAYER_MAP[layer]}.{suffix}"

    return None


def normalize_hf_key(key: str) -> str:
    if key.startswith("deberta."):
        key = key[len("deberta.") :]
    return key


def transfer_deberta_v2_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    hf = {normalize_hf_key(k): v for k, v in hf_state_dict.items()}
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        hf_name = hf_name_for(weight.path)
        if hf_name is None:
            continue
        if hf_name not in hf:
            if weight.path.startswith(_OPTIONAL_WEIGHTS):
                continue
            raise WeightMappingError(weight.path, hf_name)
        value = hf[hf_name]
        if weight.path == "conv/conv/kernel":
            # torch Conv1d (out, in, k) -> keras Conv1D (k, in, out)
            value = (
                value.detach().cpu().numpy()
                if hasattr(value, "detach")
                else np.asarray(value)
            )
            weight.assign(np.transpose(value, (2, 1, 0)))
            continue
        transfer_weights(weight.path, weight, value)
