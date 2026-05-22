import gc
import os
from typing import Dict

import numpy as np
import torch
from keras import ops
from transformers import WhisperForConditionalGeneration

from kerasformers.models.whisper import WhisperModel
from kerasformers.models.whisper.whisper_layers import WhisperAttention
from kerasformers.weight_utils.weight_transfer_torch_to_keras import (
    transfer_nested_layer_weights,
    transfer_weights,
)

HF_CHECKPOINT = {
    "whisper_tiny": "openai/whisper-tiny",
    "whisper_base": "openai/whisper-base",
    "whisper_small": "openai/whisper-small",
    "whisper_medium": "openai/whisper-medium",
    "whisper_large": "openai/whisper-large",
    "whisper_large_v2": "openai/whisper-large-v2",
    "whisper_large_v3": "openai/whisper-large-v3",
    "whisper_large_v3_turbo": "openai/whisper-large-v3-turbo",
}

DENSE_MAP = {"kernel": "weight"}
LN_MAP = {"gamma": "weight", "beta": "bias"}
EMBED_MAP = {"embeddings": "weight"}


def _strip_model_prefix(state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not any(k.startswith("model.") for k in state_dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            out[k[len("model.") :]] = v
        else:
            out[k] = v
    return out


def _transfer_encoder(encoder, state, num_layers):
    for i in (1, 2):
        conv = encoder.get_layer(f"encoder_conv{i}")
        conv.kernel.assign(np.transpose(state[f"encoder.conv{i}.weight"], (2, 1, 0)))
        transfer_weights("bias", conv.bias, state[f"encoder.conv{i}.bias"])

    encoder.get_layer("encoder_embed_positions").pos_embed.assign(
        state["encoder.embed_positions.weight"]
    )

    attns = {
        layer.name_prefix: layer
        for layer in encoder.layers
        if isinstance(layer, WhisperAttention)
    }

    for i in range(num_layers):
        kp = f"encoder_layers_{i}"
        hp = f"encoder.layers.{i}"

        sa_kp = f"{kp}_self_attn"
        transfer_nested_layer_weights(
            attns[sa_kp],
            state,
            f"{hp}.self_attn",
            name_mapping={f"{sa_kp}_": "", "kernel": "weight"},
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_self_attn_layer_norm"),
            state,
            f"{hp}.self_attn_layer_norm",
            name_mapping=LN_MAP,
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_fc1"),
            state,
            f"{hp}.fc1",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_fc2"),
            state,
            f"{hp}.fc2",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            encoder.get_layer(f"{kp}_final_layer_norm"),
            state,
            f"{hp}.final_layer_norm",
            name_mapping=LN_MAP,
        )

    transfer_nested_layer_weights(
        encoder.get_layer("encoder_layer_norm"),
        state,
        "encoder.layer_norm",
        name_mapping=LN_MAP,
    )


def _transfer_decoder(decoder, state, num_layers):
    transfer_nested_layer_weights(
        decoder.get_layer("decoder_embed_tokens"),
        state,
        "decoder.embed_tokens",
        name_mapping=EMBED_MAP,
    )

    decoder.get_layer("decoder_embed_positions").pos_embed.assign(
        state["decoder.embed_positions.weight"]
    )

    attns = {
        layer.name_prefix: layer
        for layer in decoder.layers
        if isinstance(layer, WhisperAttention)
    }

    for i in range(num_layers):
        kp = f"decoder_layers_{i}"
        hp = f"decoder.layers.{i}"

        sa_kp = f"{kp}_self_attn"
        transfer_nested_layer_weights(
            attns[sa_kp],
            state,
            f"{hp}.self_attn",
            name_mapping={f"{sa_kp}_": "", "kernel": "weight"},
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_self_attn_layer_norm"),
            state,
            f"{hp}.self_attn_layer_norm",
            name_mapping=LN_MAP,
        )

        ca_kp = f"{kp}_encoder_attn"
        transfer_nested_layer_weights(
            attns[ca_kp],
            state,
            f"{hp}.encoder_attn",
            name_mapping={f"{ca_kp}_": "", "kernel": "weight"},
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_encoder_attn_layer_norm"),
            state,
            f"{hp}.encoder_attn_layer_norm",
            name_mapping=LN_MAP,
        )

        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_fc1"),
            state,
            f"{hp}.fc1",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_fc2"),
            state,
            f"{hp}.fc2",
            name_mapping=DENSE_MAP,
        )
        transfer_nested_layer_weights(
            decoder.get_layer(f"{kp}_final_layer_norm"),
            state,
            f"{hp}.final_layer_norm",
            name_mapping=LN_MAP,
        )

    transfer_nested_layer_weights(
        decoder.get_layer("decoder_layer_norm"),
        state,
        "decoder.layer_norm",
        name_mapping=LN_MAP,
    )


def transfer_whisper_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    state = _strip_model_prefix(hf_state_dict)
    _transfer_encoder(keras_model.encoder, state, keras_model.encoder_num_layers)
    _transfer_decoder(keras_model.decoder, state, keras_model.decoder_num_layers)


def transfer_whisper_audio_classify_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    state = _strip_model_prefix(hf_state_dict)
    _transfer_encoder(keras_model.encoder, state, keras_model.encoder_num_layers)

    if keras_model.use_weighted_layer_sum:
        keras_model.get_layer("layer_weights").layer_weights.assign(
            state["layer_weights"]
        )

    projector = keras_model.get_layer("projector")
    projector.kernel.assign(np.transpose(state["projector.weight"]))
    transfer_weights("projector.bias", projector.bias, state["projector.bias"])

    classifier = keras_model.get_layer("classifier")
    classifier.kernel.assign(np.transpose(state["classifier.weight"]))
    transfer_weights("classifier.bias", classifier.bias, state["classifier.bias"])


if __name__ == "__main__":
    SLUG = {
        "whisper_tiny": "tiny",
        "whisper_base": "base",
        "whisper_small": "small",
        "whisper_medium": "medium",
        "whisper_large": "large",
        "whisper_large_v2": "largev2",
        "whisper_large_v3": "largev3",
        "whisper_large_v3_turbo": "largev3turbo",
    }

    for variant, hf_name in HF_CHECKPOINT.items():
        print(f"\n{'=' * 60}")
        print(f"Converting {hf_name}")
        print(f"{'=' * 60}")

        base = f"whisper{SLUG[variant]}_openai"
        if os.path.exists(f"{base}.weights.h5") or os.path.exists(
            f"{base}.weights.json"
        ):
            print(f"  already converted, skipping ({base})")
            continue

        print(f"[1/4] Loading {hf_name}")
        torch_model = (
            WhisperForConditionalGeneration.from_pretrained(
                hf_name, torch_dtype=torch.float32
            )
            .eval()
            .float()
        )
        state = {
            k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()
        }
        cfg = torch_model.config

        print(f"[2/4] Building Keras {variant}")
        model = WhisperModel.from_weights(variant, load_weights=False)

        print("[3/4] Transferring weights")
        transfer_whisper_weights(model, state)

        print("[4/4] Verifying parity with HF")
        np.random.seed(0)
        test_mel = np.random.randn(1, cfg.num_mel_bins, 3000).astype(np.float32)
        test_ids = np.array(
            [[cfg.decoder_start_token_id, cfg.decoder_start_token_id + 1]],
            dtype=np.int32,
        )
        keras_logits = ops.convert_to_numpy(
            model({"input_features": test_mel, "decoder_input_ids": test_ids})["logits"]
        )
        with torch.no_grad():
            hf_logits = (
                torch_model(
                    input_features=torch.from_numpy(test_mel),
                    decoder_input_ids=torch.from_numpy(test_ids),
                )
                .logits.detach()
                .cpu()
                .numpy()
            )
        diff = float(np.max(np.abs(keras_logits - hf_logits)))
        print(f"  max abs logit diff: {diff:.6e}")
        if diff > 1e-3:
            print(f"  WARNING: parity above 1e-3 (saw {diff:.6e})")

        total_params = sum(int(np.prod(w.shape)) for w in model.weights)
        total_gb = (total_params * 4) / (1024**3)
        if total_gb > 1.7:
            out_path = f"{base}.weights.json"
            model.save_weights(out_path, max_shard_size=1.7)
            print(f"Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
        else:
            out_path = f"{base}.weights.h5"
            model.save_weights(out_path)
            print(f"Saved -> {out_path} (~{total_gb:.2f} GB)")

        assert diff < 5e-3, f"{variant}: logit diff too high: {diff:.6e}"

        del torch_model, model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
