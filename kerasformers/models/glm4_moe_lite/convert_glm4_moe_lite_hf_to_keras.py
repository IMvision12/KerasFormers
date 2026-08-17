"""GLM-4.7-Flash (`glm4_moe_lite`) is the DeepSeek-V3 decoder with GLM weight names,
so its HF -> Keras conversion IS the DeepSeek-V3 conversion: the same MLA key mapping
(q_a_proj / q_b_proj / kv_a_proj_with_mqa / kv_a_layernorm / kv_b_proj / o_proj), the
same per-expert fusion, MTP-layer drop, and block-FP8 dequant. Re-export it verbatim."""

from kerasformers.models.deepseek_v3.convert_deepseek_v3_hf_to_keras import (
    WEIGHT_NAME_MAPPING,
    dequantize_fp8,
    drop_mtp_keys,
    fuse_expert_weights,
)
from kerasformers.models.deepseek_v3.convert_deepseek_v3_hf_to_keras import (
    transfer_deepseek_v3_weights as transfer_glm4_moe_lite_weights,
)

__all__ = [
    "WEIGHT_NAME_MAPPING",
    "dequantize_fp8",
    "drop_mtp_keys",
    "fuse_expert_weights",
    "transfer_glm4_moe_lite_weights",
]
