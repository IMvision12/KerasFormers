"""MLP-Mixer variant registry (timm-ported)."""

_B16 = {"patch_size": 16, "num_blocks": 12, "embed_dim": 768, "mlp_ratio": (0.5, 4.0)}
_L16 = {"patch_size": 16, "num_blocks": 24, "embed_dim": 1024, "mlp_ratio": (0.5, 4.0)}
_S16 = {"patch_size": 16, "num_blocks": 8, "embed_dim": 512, "mlp_ratio": (0.5, 4.0)}
_S32 = {"patch_size": 32, "num_blocks": 8, "embed_dim": 512, "mlp_ratio": (0.5, 4.0)}
_B32 = {"patch_size": 32, "num_blocks": 12, "embed_dim": 768, "mlp_ratio": (0.5, 4.0)}


def _v(arch, timm_id, image_size=224, num_classes=1000):
    return {
        **arch,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


MLP_MIXER_CONFIG = {
    "mixer_b16_224_goog_in21k": _v(_B16, "mixer_b16_224.goog_in21k", num_classes=21843),
    "mixer_b16_224_goog_in21k_ft_in1k": _v(_B16, "mixer_b16_224.goog_in21k_ft_in1k"),
    "mixer_b16_224_miil_in21k_ft_in1k": _v(_B16, "mixer_b16_224.miil_in21k_ft_in1k"),
    "mixer_l16_224_goog_in21k": _v(_L16, "mixer_l16_224.goog_in21k", num_classes=21843),
    "mixer_l16_224_goog_in21k_ft_in1k": _v(_L16, "mixer_l16_224.goog_in21k_ft_in1k"),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.1"
MLP_MIXER_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"}
    for variant in MLP_MIXER_CONFIG
}
