import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

from .granite_speech_model import GraniteSpeechGenerate

# Per-component Keras-fragment -> HF-fragment maps. The Keras weight path's first
# top-level segment (encoder / projector / language_model) selects which map to
# apply, so identically-named fragments in different components (e.g. the decoder's
# `attention.query` vs the q-former's `attention.attention.query`) never collide.
# Within a map, replacements are applied in order (longer/more specific first), and
# a shared set of generic param-name fixups (gamma/beta/kernel) runs last.

DECODER_MAPPING = {
    "language_model.token_embedding.embeddings": "language_model.model.embed_tokens.weight",
    "language_model.final_norm.weight": "language_model.model.norm.weight",
    "language_model.decoder_layer_": "language_model.model.layers.",
    "attention.query.lora_a": "self_attn.q_proj.lora_A.weight",
    "attention.query.lora_b": "self_attn.q_proj.lora_B.weight",
    "attention.value.lora_a": "self_attn.v_proj.lora_A.weight",
    "attention.value.lora_b": "self_attn.v_proj.lora_B.weight",
    "attention.query.kernel": "self_attn.q_proj.weight",
    "attention.key.kernel": "self_attn.k_proj.weight",
    "attention.value.kernel": "self_attn.v_proj.weight",
    "attention.output_proj.kernel": "self_attn.o_proj.weight",
    "mlp.gate.kernel": "mlp.gate_proj.weight",
    "mlp.up.kernel": "mlp.up_proj.weight",
    "mlp.down.kernel": "mlp.down_proj.weight",
}

ENCODER_MAPPING = {
    "encoder.conformer_layer_": "encoder.layers.",
    "attn.to_q.kernel": "attn.to_q.weight",
    "attn.to_kv.kernel": "attn.to_kv.weight",
    "attn.rel_pos_emb": "attn.rel_pos_emb.weight",
    "conv.depth_kernel": "conv.depth_conv.conv.weight",
    "conv.batch_norm.moving_mean": "conv.batch_norm.running_mean",
    "conv.batch_norm.moving_variance": "conv.batch_norm.running_var",
}

PROJECTOR_MAPPING = {
    "projector.layernorm": "projector.qformer.layernorm",
    "projector.qformer_layer_": "projector.qformer.encoder.layer.",
    "crossattention.dense": "crossattention.output.dense",
    "crossattention.LayerNorm": "crossattention.output.LayerNorm",
    "attention.dense": "attention.output.dense",
    "attention.LayerNorm": "attention.output.LayerNorm",
    "intermediate_query": "intermediate_query.dense",
    "output_query_dense": "output_query.dense",
    "output_query_LayerNorm": "output_query.LayerNorm",
}

GENERIC_FIXUPS = {
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}


def map_weight_name(keras_dotted):
    # The tied token embedding is tracked at the model root (it's first reached via
    # the LM head's tied access), so its dotted path is the bare
    # `token_embedding.embeddings` rather than `language_model.token_embedding...`.
    if keras_dotted == "token_embedding.embeddings":
        return "language_model.model.embed_tokens.weight"
    if keras_dotted.startswith("language_model."):
        table = DECODER_MAPPING
    elif keras_dotted.startswith("encoder."):
        table = ENCODER_MAPPING
    else:
        table = PROJECTOR_MAPPING
    name = keras_dotted
    for old, new in table.items():
        name = name.replace(old, new)
    for old, new in GENERIC_FIXUPS.items():
        name = name.replace(old, new)
    return name


def transfer_granite_speech_weights(keras_model, hf_state_dict):
    """Load a Granite Speech HF state dict (base + merged LoRA adapter) into a
    freshly built Keras model in place.

    The HF checkpoint stores the conformer encoder under ``encoder.*``, the
    Q-Former projector under ``projector.*`` and the Granite decoder under
    ``language_model.model.*``; the LoRA q/v deltas live in a separate adapter
    file under ``...self_attn.{q,v}_proj.lora_{A,B}.weight``. Linear weights are
    transposed (``transfer_weights`` handles 2D); the conformer 1x1 convs (stored
    ``(out, in, 1)``) and depthwise conv (``(channels, 1, kernel)``) are reshaped
    here, and the q-former ``query`` / projector ``query`` parameters are copied
    verbatim.
    """
    if not keras_model.built or not keras_model.weights:
        keras_model.build_dummy()

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = map_weight_name(weight.path.split("/", 1)[1].replace("/", "."))

        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        torch_weight = np.asarray(hf_state_dict[name])

        # Weights assigned verbatim / with an explicit reshape (transfer_weights
        # would mis-transpose these 2D tensors as generic Dense kernels):
        #   - learned query tokens + Shaw relative-position table: same shape.
        #   - depthwise Conv1d (channels, 1, kernel) -> (kernel, channels).
        #   - pointwise Conv1d (out, in, 1) -> Dense kernel (in, out).
        if weight.path.endswith("rel_pos_emb") or weight.path.endswith("query"):
            weight.assign(torch_weight)
            continue
        if "depth_kernel" in weight.path:
            weight.assign(torch_weight[:, 0, :].T)
            continue
        if (
            "conv/up_conv" in weight.path or "conv/down_conv" in weight.path
        ) and weight.path.endswith("kernel"):
            weight.assign(torch_weight[:, :, 0].T)
            continue

        transfer_weights(weight.path, weight, torch_weight)


def build_merged_state_dict(hf_model_id, dtype="float32"):
    """Return a flat ``{name: np.ndarray}`` of the base model weights merged with
    the LoRA adapter tensors (kept as separate ``lora_A`` / ``lora_B`` keys so the
    Keras LoRA branch loads them directly)."""
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file

    files = list_repo_files(hf_model_id)
    state = {}
    # Base safetensors shards.
    shard_files = sorted(
        f for f in files if f.startswith("model") and f.endswith(".safetensors")
    )
    for shard in shard_files:
        path = hf_hub_download(hf_model_id, shard)
        for k, v in load_file(path).items():
            state[k] = v.to(torch.float32).cpu().numpy()
    # LoRA adapter.
    if "adapter_model.safetensors" in files:
        ap = hf_hub_download(hf_model_id, "adapter_model.safetensors")
        for k, v in load_file(ap).items():
            k = k.replace("base_model.model.", "")
            state[k] = v.to(torch.float32).cpu().numpy()
    return state


if __name__ == "__main__":
    import gc
    import math

    from keras import ops

    from .config import GRANITE_SPEECH_HF_IDS

    VARIANT = "granite-speech-3.3-2b"
    HF_ID = GRANITE_SPEECH_HF_IDS[VARIANT]
    # Sharded index (+ shards) so the user uploads them under the release tag.
    OUT = f"C:/Users/gites/Desktop/code/v1_weights/{VARIANT.replace('-', '_')}.weights.json"

    print(f"[1/4] Building merged state dict from {HF_ID}")
    state = build_merged_state_dict(HF_ID)

    print("[2/4] Building Keras model + transferring weights")
    model = GraniteSpeechGenerate.from_weights(VARIANT, load_weights=False)
    transfer_granite_speech_weights(model, state)

    print("[3/4] Sanity forward")
    frames = 4 * model.window_size
    nblocks = math.ceil(frames / model.window_size)
    n_audio = nblocks * (model.window_size // model.downsample_rate)
    out = model(
        {
            "input_ids": np.array(
                [[1] + [model.audio_token_id] * n_audio + [2]], dtype="int64"
            ),
            "input_features": np.zeros(
                (1, frames, model.encoder_input_dim), dtype="float32"
            ),
        }
    )
    print("  logits", tuple(ops.convert_to_numpy(out["logits"]).shape))

    print(f"[4/4] Saving sharded weights to {OUT}")
    # max_shard_size is in GB (a float). 2.0 keeps shards comfortably under the
    # GitHub-release 2 GB per-asset cap for the ~9 GB 2b checkpoint.
    model.save_weights(OUT, max_shard_size=2.0)
    del state
    gc.collect()
