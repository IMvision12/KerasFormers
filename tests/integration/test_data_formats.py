import os

import keras
import pytest
from keras import ops

from tests.base.model_test_registry import (
    MODEL_TEST_CONFIGS,
    create_test_input,
    import_model_class,
)

BACKEND = os.environ.get("KERAS_BACKEND", "torch")
MODEL_IDS = list(MODEL_TEST_CONFIGS.keys())

# Models that don't support runtime channels_first/channels_last switching:
# - Whisper* / Speech2Text* / Moonshine*: audio models, no spatial image
#   dim; the channels_first conversion doesn't apply.
# - MaskFormerUniversalSegment / Mask2FormerUniversalSegment: the HF-aligned Swin backbone
#   port works in channels_last only (the conv → reshape → flatten path
#   assumes (B, H, W, C)). HF MaskFormer / Mask2Former checkpoints are
#   only released for channels_last; supporting channels_first would
#   require an alternate code path in the backbone.
SKIP_DATA_FORMAT = {
    "WhisperModel",
    "WhisperSpeechToText",
    "WhisperAudioClassify",
    "Speech2TextModel",
    "Speech2TextSpeechToText",
    "MoonshineModel",
    "MoonshineSpeechToText",
    "GraniteSpeechModel",
    "GraniteSpeechGenerate",
    "GraniteSpeechPlusModel",
    "GraniteSpeechPlusGenerate",
    "MaskFormerUniversalSegment",
    "Mask2FormerUniversalSegment",
    # Qwen-VL inputs are pre-patchified (no spatial axes) -> layout-agnostic.
    "Qwen2VLModel",
    "Qwen2_5VLModel",
    "Qwen3VLModel",
    "Qwen2VLGenerate",
    "Qwen2_5VLGenerate",
    "Qwen3VLGenerate",
    # Text LLMs are token-id only -> no image data format.
    "Qwen2Model",
    "Qwen3Model",
    "Qwen3_5Model",
    "Qwen2Generate",
    "Qwen3Generate",
    "Qwen3_5Generate",
}


def _adapt_input_shape_for_format(init_kwargs, data_format):
    kwargs = init_kwargs.copy()
    if data_format == "channels_first" and "input_shape" in kwargs:
        h, w, c = kwargs["input_shape"]
        kwargs["input_shape"] = (c, h, w)
    if data_format == "channels_first" and "image_size" in kwargs:
        spec = kwargs["image_size"]
        if isinstance(spec, (tuple, list)) and len(spec) == 3:
            h, w, c = spec
            kwargs["image_size"] = (c, h, w)
    return kwargs


def _transpose_input(input_data, data_format):
    if data_format != "channels_first":
        return input_data
    if isinstance(input_data, dict):
        result = {}
        for k, v in input_data.items():
            if k in ("pixel_values", "images") and len(v.shape) == 4:
                result[k] = ops.transpose(v, (0, 3, 1, 2))
            else:
                result[k] = v
        return result
    if len(input_data.shape) == 4:
        return ops.transpose(input_data, (0, 3, 1, 2))
    return input_data


@pytest.mark.data_format
@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_channels_last(model_name):
    if model_name in SKIP_DATA_FORMAT:
        pytest.skip(f"{model_name} doesn't support data format switching")

    original = keras.config.image_data_format()
    try:
        keras.config.set_image_data_format("channels_last")
        config = MODEL_TEST_CONFIGS[model_name]
        model_cls = import_model_class(config)
        model = model_cls(**config["init_kwargs"])
        input_data = create_test_input(config)
        output = model(input_data)

        if isinstance(output, dict):
            for key, value in output.items():
                assert not bool(ops.any(ops.isnan(value))), (
                    f"{model_name}[{key}] has NaNs in channels_last"
                )
        elif isinstance(output, (list, tuple)):
            for i, value in enumerate(output):
                assert not bool(ops.any(ops.isnan(value))), (
                    f"{model_name}[{i}] has NaNs in channels_last"
                )
        else:
            assert not bool(ops.any(ops.isnan(output))), (
                f"{model_name} has NaNs in channels_last"
            )
    finally:
        keras.config.set_image_data_format(original)


@pytest.mark.data_format
@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_channels_first(model_name):
    if model_name in SKIP_DATA_FORMAT:
        pytest.skip(f"{model_name} doesn't support data format switching")

    if BACKEND == "tensorflow":
        try:
            import tensorflow as tf

            if not tf.config.list_physical_devices("GPU"):
                pytest.skip("TF channels_first conv2d requires GPU (cuDNN)")
        except ImportError:
            pytest.skip("TensorFlow not installed")

    original = keras.config.image_data_format()
    try:
        keras.config.set_image_data_format("channels_first")
        config = MODEL_TEST_CONFIGS[model_name]
        model_cls = import_model_class(config)
        adapted_kwargs = _adapt_input_shape_for_format(
            config["init_kwargs"], "channels_first"
        )
        model = model_cls(**adapted_kwargs)
        input_data = create_test_input(config)
        input_data = _transpose_input(input_data, "channels_first")
        output = model(input_data)

        if isinstance(output, dict):
            for key, value in output.items():
                assert not bool(ops.any(ops.isnan(value))), (
                    f"{model_name}[{key}] has NaNs in channels_first"
                )
        elif isinstance(output, (list, tuple)):
            for i, value in enumerate(output):
                assert not bool(ops.any(ops.isnan(value))), (
                    f"{model_name}[{i}] has NaNs in channels_first"
                )
        else:
            assert not bool(ops.any(ops.isnan(output))), (
                f"{model_name} has NaNs in channels_first"
            )
    finally:
        keras.config.set_image_data_format(original)
