import keras

from kerasformers.models.qwen2_vl.qwen2_vl_processor import Qwen2VLProcessor


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLProcessor(Qwen2VLProcessor):
    """Qwen3-VL image+text processor: identical to :class:`Qwen2VLProcessor`
    except ``patch_size`` defaults to 16."""

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
