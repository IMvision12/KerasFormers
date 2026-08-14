import keras

from kerasformers.models.qwen3_vl.qwen3_vl_processor import Qwen3VLProcessor

from .qwen3_vl_moe_tokenizer import Qwen3VLMoeTokenizer
from .qwen3_vl_moe_video_processor import Qwen3VLMoeVideoProcessor


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3VLMoeProcessor(Qwen3VLProcessor):
    """Qwen3-VL-MoE image/video+text processor: identical to :class:`Qwen3VLProcessor`
    (16px patch, ChatML + image-pad expansion) with the Qwen3-VL-MoE tokenizer and
    video processor. The image processor is reused from Qwen2-VL."""

    TOKENIZER_CLS = Qwen3VLMoeTokenizer
    video_processor_cls = Qwen3VLMoeVideoProcessor

    def __init__(self, hf_id="Qwen/Qwen3-VL-30B-A3B-Instruct", **kwargs):
        super().__init__(hf_id=hf_id, **kwargs)
