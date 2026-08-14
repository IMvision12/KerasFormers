import keras

from kerasformers.models.qwen2_vl.qwen2_vl_processor import Qwen2VLProcessor

from .qwen3_5_moe_tokenizer import Qwen3_5MoeTokenizer
from .qwen3_5_moe_video_processor import Qwen3_5MoeVideoProcessor


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3_5MoeProcessor(Qwen2VLProcessor):
    """Qwen3.5-MoE image/video+text processor.

    Like :class:`Qwen2VLProcessor` but with a 16px patch, the Qwen3.5-MoE tokenizer,
    and the Qwen3.5-MoE (clip-level) video processor. The image processor is reused
    from Qwen2-VL: only the patch size differs (16 vs 14).
    """

    TOKENIZER_CLS = Qwen3_5MoeTokenizer
    video_processor_cls = Qwen3_5MoeVideoProcessor

    def __init__(
        self,
        hf_id="Qwen/Qwen3.5-35B-A3B-Instruct",
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
