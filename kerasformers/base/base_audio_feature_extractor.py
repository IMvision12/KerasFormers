from kerasformers.base.base_preprocessing import BasePreprocessingLayer


class BaseAudioFeatureExtractor(BasePreprocessingLayer):
    """Abstract base for kerasformers audio feature extractors.

    Subclasses implement ``call(raw_speech, ...)`` returning the spectrogram /
    feature tensor. The loading API (``from_weights`` / ``from_release`` /
    ``from_hf``) and the ``__call__`` -> ``call`` forwarder are inherited from
    :class:`BasePreprocessingLayer`. Concrete subclasses define their own
    constructor kwargs (sampling rate, FFT size, mel bin count, chunk length, …)
    and ``get_config`` payload — the base bakes in no defaults.
    """

    def call(self, raw_speech, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `call(raw_speech, ...)`."
        )
