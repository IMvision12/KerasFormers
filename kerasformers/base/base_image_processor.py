from kerasformers.base.base_processor import BasePreprocessingLayer


class BaseImageProcessor(BasePreprocessingLayer):
    """Abstract base for kerasformers image preprocessors.

    Subclasses implement ``call(images)`` returning the model-ready pixel tensor
    (or a dict that includes one). The loading API (``from_weights`` /
    ``from_release`` / ``from_hf``) and the ``__call__`` -> ``call`` forwarder are
    inherited from :class:`BasePreprocessingLayer`. Concrete subclasses define
    their own constructor kwargs (resolution, normalization stats, interpolation
    mode, patch size, …) and ``get_config`` payload — the base bakes in no
    defaults.
    """

    def call(self, images):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `call(images)`."
        )
