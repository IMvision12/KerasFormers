import gc
import sys

import keras
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from kerasformers.models.metaclip2 import MetaClip2ZeroShotClassify
from kerasformers.weight_utils.custom_exception import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from kerasformers.weight_utils.weight_split_torch_and_keras import split_model_weights
from kerasformers.weight_utils.weight_transfer_torch_to_keras import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)

weight_name_mapping = {
    "_": ".",
    "vision.model": "vision_model",
    "text.model": "text_model",
    "conv": "embeddings.patch_embedding",
    "class.embedding": "class_embedding",
    "pos.embed": "position_embedding.weight",
    "vision_model.layernorm.1": "vision_model.pre_layrnorm",
    "text_model.encoder": "text_model.encoder.layers",
    "vision_model.encoder": "vision_model.encoder.layers",
    "text_model.layernorm": "text_model.final_layer_norm",
    "layernorm.1": "layer_norm1",
    "layernorm.2": "layer_norm2",
    "vision_model.layer_norm2": "vision_model.post_layernorm",
    "text.projection": "text_projection",
    "visual.projection": "visual_projection",
    "logit_scale_logit_scale": "logit_scale",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
}

attn_name_replace = {
    "text.model": "text_model",
    "vision.model": "vision_model",
    "encoder": "encoder.layers",
    "attn": "self_attn",
    "q.proj": "q_proj",
    "k.proj": "k_proj",
    "v.proj": "v_proj",
    "out.proj": "out_proj",
}

HF_REPO = {
    "metaclip2_worldwide_s16_224": "facebook/metaclip-2-worldwide-s16",
    "metaclip2_worldwide_s16_384": "facebook/metaclip-2-worldwide-s16-384",
    "metaclip2_worldwide_m16_224": "facebook/metaclip-2-worldwide-m16",
    "metaclip2_worldwide_m16_384": "facebook/metaclip-2-worldwide-m16-384",
    "metaclip2_worldwide_b16_224": "facebook/metaclip-2-worldwide-b16",
    "metaclip2_worldwide_b16_384": "facebook/metaclip-2-worldwide-b16-384",
    "metaclip2_worldwide_b32_224": "facebook/metaclip-2-worldwide-b32",
    "metaclip2_worldwide_b32_384": "facebook/metaclip-2-worldwide-b32-384",
    "metaclip2_worldwide_l14_224": "facebook/metaclip-2-worldwide-l14",
    "metaclip2_worldwide_huge_quickgelu": "facebook/metaclip-2-worldwide-huge-quickgelu",
    "metaclip2_worldwide_huge_378": "facebook/metaclip-2-worldwide-huge-378",
    "metaclip2_worldwide_giant_224": "facebook/metaclip-2-worldwide-giant",
    "metaclip2_worldwide_giant_378": "facebook/metaclip-2-worldwide-giant-378",
    "metaclip2_mt5_worldwide_s16_224": "facebook/metaclip-2-mt5-worldwide-s16",
    "metaclip2_mt5_worldwide_m16_224": "facebook/metaclip-2-mt5-worldwide-m16",
    "metaclip2_mt5_worldwide_b32_224": "facebook/metaclip-2-mt5-worldwide-b32",
}


def transfer_metaclip2_weights(keras_model, hf_state_dict):
    trainable_k, non_trainable_k = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable_k + non_trainable_k,
        total=len(trainable_k + non_trainable_k),
        desc="Converting MetaClip2",
        unit="weight",
    ):
        torch_weight_name = keras_weight_name
        for a, b in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(a, b)

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, hf_state_dict, attn_name_replace
            )
            continue

        if keras_weight_name == "text_model_embedding_embeddings":
            if "token_embedding" in keras_weight.path:
                keras_weight.assign(
                    hf_state_dict["text_model.embeddings.token_embedding.weight"]
                )
                continue
            if "positional_embedding" in keras_weight.path:
                keras_weight.assign(
                    hf_state_dict["text_model.embeddings.position_embedding.weight"]
                )
                continue

        if keras_weight_name == "logit_scale_logit_scale":
            keras_weight.assign(hf_state_dict["logit_scale"])
            continue

        if keras_weight_name == "vision_model_embeddings_pos_embed":
            pos = hf_state_dict["vision_model.embeddings.position_embedding.weight"]
            keras_weight.assign(np.expand_dims(pos, 0))
            continue

        if torch_weight_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)
        tw = hf_state_dict[torch_weight_name]
        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_weight_name, tw
        ):
            raise WeightShapeMismatchError(
                keras_weight_name, keras_weight.shape, torch_weight_name, tw.shape
            )
        transfer_weights(keras_weight_name, keras_weight, tw)


def transfer_metaclip2_image_classify_weights(keras_model, hf_state_dict):
    has_classifier = (
        "classifier.weight" in hf_state_dict and "classifier.bias" in hf_state_dict
    )
    trainable_k, non_trainable_k = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in trainable_k + non_trainable_k:
        if keras_weight_name in ("classifier_kernel", "classifier_bias"):
            if not has_classifier:
                continue
            if "kernel" in keras_weight.path:
                keras_weight.assign(np.transpose(hf_state_dict["classifier.weight"]))
            else:
                keras_weight.assign(hf_state_dict["classifier.bias"])
            continue

        torch_weight_name = keras_weight_name
        for a, b in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(a, b)

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, hf_state_dict, attn_name_replace
            )
            continue

        if keras_weight_name == "vision_model_embeddings_pos_embed":
            pos = hf_state_dict["vision_model.embeddings.position_embedding.weight"]
            keras_weight.assign(np.expand_dims(pos, 0))
            continue

        if torch_weight_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)
        tw = hf_state_dict[torch_weight_name]
        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_weight_name, tw
        ):
            raise WeightShapeMismatchError(
                keras_weight_name, keras_weight.shape, torch_weight_name, tw.shape
            )
        transfer_weights(keras_weight_name, keras_weight, tw)


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "metaclip2_worldwide_b32_224"
    hf_id = HF_REPO[variant]

    print(f"Converting {variant}  <-  {hf_id}")

    hf_model = AutoModel.from_pretrained(hf_id).eval()
    state = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}

    keras_model = MetaClip2ZeroShotClassify.from_weights(variant, load_weights=False)
    transfer_metaclip2_weights(keras_model, state)

    total_params = sum(int(np.prod(w.shape)) for w in keras_model.weights)
    total_gb = (total_params * 4) / (1024**3)

    del state
    gc.collect()

    if total_gb <= 5.0:
        ctx = keras_model.context_length
        vocab = keras_model.vocab_size
        eos = keras_model.eos_token_id
        ishape = keras_model.input_image_shape
        if keras.config.image_data_format() == "channels_first":
            img_h, img_w = ishape[1], ishape[2]
        else:
            img_h, img_w = ishape[0], ishape[1]

        rng = np.random.default_rng(0)
        pixel = rng.standard_normal((2, img_h, img_w, 3)).astype(np.float32)
        token_ids = rng.integers(0, vocab - 1, size=(2, ctx)).astype(np.int32)
        token_ids[:, -1] = eos  # EOS id at the pooled position
        attn = np.ones((2, ctx), dtype=np.int32)

        with torch.no_grad():
            hf_out = hf_model(
                pixel_values=torch.from_numpy(pixel.transpose(0, 3, 1, 2)),
                input_ids=torch.from_numpy(token_ids.astype(np.int64)),
                attention_mask=torch.from_numpy(attn.astype(np.int64)),
            )
            hf_logits = hf_out.logits_per_image.cpu().numpy()
            scale = hf_model.logit_scale.exp().item()

        k_out = keras_model(
            {"images": pixel, "token_ids": token_ids, "padding_mask": attn},
            training=False,
        )
        k_logits = keras.ops.convert_to_numpy(k_out["image_logits"])

        logits_diff = float(np.abs(hf_logits - k_logits).max())
        cosine_diff = logits_diff / (scale + 1e-8)
        print(
            f"  Max logits diff: {logits_diff:.6f}  (cosine-level: {cosine_diff:.2e})"
        )
        if cosine_diff > 1e-2:
            raise ValueError(
                f"{variant}: equivalence check failed "
                f"(logits diff {logits_diff:.4f}, cosine {cosine_diff:.2e})"
            )
    else:
        print(
            f"  Equivalence check skipped (~{total_gb:.1f} GB model exceeds "
            f"RAM budget; weights validated by name-based mapping)"
        )

    if total_gb > 1.7:
        out_path = f"{variant}.weights.json"
        keras_model.save_weights(out_path, max_shard_size=1.9)
        print(f"  Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
    else:
        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path} (~{total_gb:.2f} GB)")

    del keras_model, hf_model
    keras.backend.clear_session()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
