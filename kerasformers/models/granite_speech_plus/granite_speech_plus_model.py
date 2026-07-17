import keras

from kerasformers.models.granite_speech.granite_speech_model import (
    GraniteSpeechGenerate,
    GraniteSpeechModel,
)

from .granite_speech_plus_config import (
    GRANITE_SPEECH_PLUS_CONFIG,
    GRANITE_SPEECH_PLUS_WEIGHTS_URLS,
)


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechPlusModel(GraniteSpeechModel):
    """GraniteSpeechPlus backbone: :class:`GraniteSpeechModel` whose conformer CTC
    encoder concatenates a subset of intermediate layer outputs
    (``cat_hidden_layers``) with its final output before the projector (so the
    projector's ``encoder_hidden_size`` becomes
    ``encoder_hidden_dim * (len(cat_hidden_layers) + 1)``). All layers, fusion, the
    LoRA adapter and the weight transfer are reused from ``granite_speech``; this
    variant only points at the Plus config + release weights."""

    HF_MODEL_TYPE = "granite_speech_plus"
    BASE_MODEL_CONFIG = GRANITE_SPEECH_PLUS_CONFIG
    BASE_WEIGHT_CONFIG = GRANITE_SPEECH_PLUS_WEIGHTS_URLS


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechPlusGenerate(GraniteSpeechGenerate):
    """GraniteSpeechPlus with an LM head + fast ``.generate()`` (audio+text -> text)
    the Plus variant of :class:`GraniteSpeechGenerate`."""

    HF_MODEL_TYPE = "granite_speech_plus"
    BASE_MODEL_CONFIG = GRANITE_SPEECH_PLUS_CONFIG
    BASE_WEIGHT_CONFIG = GRANITE_SPEECH_PLUS_WEIGHTS_URLS
