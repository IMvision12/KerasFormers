import keras

from kerasformers.models.deepseek_v3.deepseek_v3_model import (
    DeepseekV3Model,
    DeepseekV3TextGenerate,
)

from .glm4_moe_lite_config import Glm4MoeLiteConfig


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4MoeLiteModel(DeepseekV3Model):
    """GLM-4.7-Flash backbone (``glm4_moe_lite``): the DeepSeek-V3 decoder (MLA +
    aux-loss-free DeepSeekMoE) under GLM naming, i.e. GLM-5 without the DSA indexer.

    The architecture is byte-for-byte :class:`DeepseekV3Model`, so this subclasses it
    verbatim and only swaps the loading identity (``model_type`` / ``config_class``).
    Its weights convert with the DeepSeek-V3 mapping (same MLA / MoE key names).
    """

    HF_MODEL_TYPE = "glm4_moe_lite"
    config_class = Glm4MoeLiteConfig
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_glm4_moe_lite_hf_to_keras import transfer_glm4_moe_lite_weights

        transfer_glm4_moe_lite_weights(keras_model, hf_state_dict)


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4MoeLiteTextGenerate(Glm4MoeLiteModel, DeepseekV3TextGenerate):
    """GLM-4.7-Flash with an LM head + fast ``.generate()``.

    Inherits the whole MLA generation path (LM head, ``build_cache`` /
    ``call_with_cache``: expanded per-head k/v cache) from
    :class:`~kerasformers.models.deepseek_v3.deepseek_v3_model.DeepseekV3TextGenerate`;
    only the stop tokens differ.
    """

    # GLM-4.7-Flash generation_config eos ids: <|user|> / <|observation|> / <|endoftext|>.
    eos_token_id = (154820, 154827, 154829)
