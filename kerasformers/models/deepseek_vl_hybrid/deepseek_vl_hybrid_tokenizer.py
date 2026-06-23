import keras

from kerasformers.models.deepseek_vl.deepseek_vl_tokenizer import DeepseekVLTokenizer

from .config import DEEPSEEK_VL_HYBRID_TOKENIZER_URLS


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekVLHybridTokenizer(DeepseekVLTokenizer):
    """DeepSeek-VL Hybrid (7B) tokenizer.

    Byte-for-byte the same BPE tokenizer as the 1.3B
    :class:`~kerasformers.models.deepseek_vl.DeepseekVLTokenizer` (the whole
    DeepSeek-VL family shares one vocab — ``<image_placeholder>`` id 100015,
    vocab 100016); only the release variant URLs differ, so this just re-points
    ``TOKENIZER_URLS`` / ``DEFAULT_VARIANT`` at the 7B release.
    """

    TOKENIZER_URLS = DEEPSEEK_VL_HYBRID_TOKENIZER_URLS
    DEFAULT_VARIANT = "deepseek-vl-7b-chat"
