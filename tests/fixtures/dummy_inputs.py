import numpy as np
from keras import ops


def qwen_vl_input(
    batch_size=2, grid=(1, 4, 4), patch_dim=1176, image_token_id=4, merge=2
):
    """Multimodal input for the Qwen-VL models: pre-patchified image patches +
    input_ids with the right number of image-placeholder tokens per row."""
    t, gh, gw = grid
    n_patches = t * gh * gw
    n_merged = t * (gh // merge) * (gw // merge)
    seq = n_merged + 2
    ids = np.zeros((batch_size, seq), dtype="int32")
    ids[:, 0] = 10
    ids[:, 1 : 1 + n_merged] = image_token_id
    ids[:, -1] = 11
    return {
        "input_ids": ops.convert_to_tensor(ids),
        "pixel_values": ops.ones((batch_size * n_patches, patch_dim)),
        "image_grid_thw": ops.convert_to_tensor(
            np.tile(np.array(grid, dtype="int32"), (batch_size, 1))
        ),
    }


def qwen_text_input(batch_size=2, seq_len=6, vocab_size=128):
    """Token-id input for the pure-text Qwen LLMs (Qwen2 / Qwen3 / Qwen3.5)."""
    ids = np.tile(np.arange(seq_len, dtype="int32") % vocab_size, (batch_size, 1))
    return {"input_ids": ops.convert_to_tensor(ids)}


def backbone_input(batch_size=2, spatial=32, channels=3):
    return ops.ones((batch_size, spatial, spatial, channels))


def detection_input(batch_size=2, spatial=32, channels=3):
    return ops.ones((batch_size, spatial, spatial, channels))


def segmentation_input(batch_size=2, spatial=32, channels=3):
    return ops.ones((batch_size, spatial, spatial, channels))


def clip_input(batch_size=2, image_size=64, max_seq_len=77):
    return {
        "images": ops.ones((batch_size, image_size, image_size, 3)),
        "token_ids": ops.ones((batch_size, max_seq_len), dtype="int32"),
        "padding_mask": ops.ones((batch_size, max_seq_len), dtype="int32"),
    }


def siglip_input(batch_size=2, image_size=64, max_seq_len=64):
    return {
        "images": ops.ones((batch_size, image_size, image_size, 3)),
        "token_ids": ops.ones((batch_size, max_seq_len), dtype="int32"),
        "padding_mask": ops.ones((batch_size, max_seq_len), dtype="int32"),
    }


def sam_input(batch_size=2, image_size=64, num_prompts=1, num_points=1):
    return {
        "pixel_values": ops.ones((batch_size, image_size, image_size, 3)),
        "input_points": ops.ones(
            (batch_size, num_prompts, num_points, 2), dtype="float32"
        ),
        "input_labels": ops.ones((batch_size, num_prompts, num_points), dtype="int32"),
    }


def owlvit_input(batch_size=2, image_size=64, max_seq_len=16, num_queries=2):
    return {
        "pixel_values": ops.ones((batch_size, image_size, image_size, 3)),
        "input_ids": ops.ones((batch_size * num_queries, max_seq_len), dtype="int32"),
    }


def whisper_input(batch_size=2, num_mel_bins=80, mel_length=40, decoder_seq_len=5):
    return {
        "input_features": ops.ones((batch_size, num_mel_bins, mel_length)),
        "decoder_input_ids": ops.zeros((batch_size, decoder_seq_len), dtype="int32"),
    }


def whisper_audio_input(batch_size=2, num_mel_bins=80, mel_length=40):
    return ops.ones((batch_size, num_mel_bins, mel_length))


def speech2text_input(batch_size=2, num_mel_bins=80, feat_length=40, decoder_seq_len=5):
    # Speech2Text fbank features are (batch, time, num_mel_bins) - the transpose
    # of Whisper's (batch, num_mel_bins, time) layout.
    return {
        "input_features": ops.ones((batch_size, feat_length, num_mel_bins)),
        "decoder_input_ids": ops.zeros((batch_size, decoder_seq_len), dtype="int32"),
    }
