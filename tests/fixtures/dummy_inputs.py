from keras import ops


def backbone_input(batch_size=2, spatial=32, channels=3):
    return ops.ones((batch_size, spatial, spatial, channels))


def detection_input(batch_size=2, spatial=32, channels=3):
    return ops.ones((batch_size, spatial, spatial, channels))


def segmentation_input(batch_size=2, spatial=32, channels=3):
    return ops.ones((batch_size, spatial, spatial, channels))


def clip_input(batch_size=2, image_size=64, context_length=77):
    return {
        "images": ops.ones((batch_size, image_size, image_size, 3)),
        "token_ids": ops.ones((batch_size, context_length), dtype="int32"),
        "padding_mask": ops.ones((batch_size, context_length), dtype="int32"),
    }


def siglip_input(batch_size=2, image_size=64, context_length=64):
    return {
        "images": ops.ones((batch_size, image_size, image_size, 3)),
        "token_ids": ops.ones((batch_size, context_length), dtype="int32"),
        "padding_mask": ops.ones((batch_size, context_length), dtype="int32"),
    }


def sam_input(batch_size=2, image_size=64, num_prompts=1, num_points=1):
    return {
        "pixel_values": ops.ones((batch_size, image_size, image_size, 3)),
        "input_points": ops.ones(
            (batch_size, num_prompts, num_points, 2), dtype="float32"
        ),
        "input_labels": ops.ones((batch_size, num_prompts, num_points), dtype="int32"),
    }


def owlvit_input(batch_size=2, image_size=64, context_length=16, num_queries=2):
    return {
        "pixel_values": ops.ones((batch_size, image_size, image_size, 3)),
        "input_ids": ops.ones(
            (batch_size * num_queries, context_length), dtype="int32"
        ),
    }


def whisper_input(batch_size=2, num_mel_bins=80, mel_length=40, decoder_seq_len=5):
    return {
        "input_features": ops.ones((batch_size, num_mel_bins, mel_length)),
        "decoder_input_ids": ops.zeros((batch_size, decoder_seq_len), dtype="int32"),
    }


def whisper_audio_input(batch_size=2, num_mel_bins=80, mel_length=40):
    return ops.ones((batch_size, num_mel_bins, mel_length))
