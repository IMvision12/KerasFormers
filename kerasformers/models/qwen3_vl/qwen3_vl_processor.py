import keras

from kerasformers.models.qwen2_vl.qwen2_vl_processor import Qwen2VLProcessor

from .qwen3_vl_video_processor import Qwen3VLVideoProcessor


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLProcessor(Qwen2VLProcessor):
    """Qwen3-VL image/video+text processor: like :class:`Qwen2VLProcessor` but
    with a 16px patch and the Qwen3-VL video processor (``[0.5]*3`` normalization
    and a clip-level resize budget)."""

    video_processor_cls = Qwen3VLVideoProcessor

    def __init__(
        self,
        hf_id="Qwen/Qwen3-VL-2B-Instruct",
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        **kwargs,
    ):
        super().__init__(
            hf_id=hf_id,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            **kwargs,
        )
