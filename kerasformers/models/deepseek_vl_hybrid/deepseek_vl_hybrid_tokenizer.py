import keras

from kerasformers.models.deepseek_vl.deepseek_vl_tokenizer import DeepseekVLTokenizer


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekVLHybridTokenizer(DeepseekVLTokenizer):
    """DeepSeek-VL Hybrid (7B) tokenizer.

    Byte-for-byte the same BPE tokenizer as the 1.3B
    :class:`~kerasformers.models.deepseek_vl.DeepseekVLTokenizer` (the whole
    DeepSeek-VL family shares one vocab: ``<image_placeholder>`` id 100015,
    vocab 100016); only the default variant differs so ``kerasformers/<variant>``
    resolves to the 7B Hub repo.
    """

    DEFAULT_VARIANT = "deepseek_vl_7b_chat"
