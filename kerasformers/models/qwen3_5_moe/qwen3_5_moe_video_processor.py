import keras

from kerasformers.models.qwen3_vl.qwen3_vl_video_processor import Qwen3VLVideoProcessor


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3_5MoeVideoProcessor(Qwen3VLVideoProcessor):
    """Qwen3.5-MoE video processor: identical to :class:`Qwen3VLVideoProcessor`
    (16px patch, ``[0.5]*3`` normalization, clip-level frame-count-aware resize)."""
