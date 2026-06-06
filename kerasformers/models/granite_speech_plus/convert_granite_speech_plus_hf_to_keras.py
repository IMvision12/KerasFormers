from kerasformers.models.granite_speech.convert_granite_speech_hf_to_keras import (
    build_merged_state_dict,
    transfer_granite_speech_weights,
)

from .granite_speech_plus_model import GraniteSpeechPlusGenerate

# GraniteSpeechPlus shares GraniteSpeech's weight layout (the cat_hidden_layers
# concatenation is a forward-time op, no extra weights), so the transfer + merged
# state-dict builder are reused verbatim; only the variant + model class differ.

if __name__ == "__main__":
    import gc
    import math

    import numpy as np
    from keras import ops

    from .config import GRANITE_SPEECH_PLUS_HF_IDS

    VARIANT = "granite-speech-4.1-2b-plus"
    HF_ID = GRANITE_SPEECH_PLUS_HF_IDS[VARIANT]
    # Sharded index (+ shards) so the user uploads them under the release tag.
    OUT = f"C:/Users/gites/Desktop/code/v1_weights/{VARIANT.replace('-', '_')}.weights.json"

    print(f"[1/4] Building merged state dict from {HF_ID}")
    state = build_merged_state_dict(HF_ID)

    print("[2/4] Building Keras model + transferring weights")
    model = GraniteSpeechPlusGenerate.from_weights(VARIANT, load_weights=False)
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
    model.save_weights(OUT, max_shard_size=2.0)
    del state
    gc.collect()
