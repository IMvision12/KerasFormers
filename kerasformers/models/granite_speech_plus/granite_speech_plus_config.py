from kerasformers.models.granite_speech.granite_speech_config import (
    GraniteSpeechConfig,
)


class GraniteSpeechPlusConfig(GraniteSpeechConfig):
    r"""Configuration for GraniteSpeechPlus ([`GraniteSpeechPlusModel`] /
    [`GraniteSpeechPlusGenerate`]).

    Same architecture as [`GraniteSpeechConfig`], with `cat_hidden_layers` set: the
    conformer CTC encoder concatenates the listed intermediate layer outputs with
    its final output before the projector. This config inherits every field from
    [`GraniteSpeechConfig`]; the single released variant sets its dimensions and
    `cat_hidden_layers` via the recipe. Fields serialize flat to a repo's
    `kf_config.json` (declaring the canonical [`GraniteSpeechPlusGenerate`]).

    Examples:

    ```python
    >>> from kerasformers.models.granite_speech_plus import (
    ...     GraniteSpeechPlusConfig, GraniteSpeechPlusGenerate)

    >>> configuration = GraniteSpeechPlusConfig()
    >>> model = GraniteSpeechPlusGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "granite_speech_plus"
