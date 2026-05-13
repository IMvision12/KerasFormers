"""MobileViTV2 variant registry (timm-ported).

Variant ids follow timm naming: ``<arch>_<recipe>_<train_dataset>``.
``timm_id`` is the corresponding timm repo on HuggingFace Hub, used
by both the offline conversion script and the runtime ``timm:`` path
on :meth:`MobileViTV2.from_timm`.
"""


def _v(multiplier, timm_id, image_size=256, num_classes=1000):
    return {
        "multiplier": multiplier,
        "timm_id": timm_id,
        "image_size": image_size,
        "num_classes": num_classes,
    }


MOBILEVITV2_CONFIG = {
    "mobilevitv2_050_cvnets_in1k": _v(0.5, "mobilevitv2_050.cvnets_in1k"),
    "mobilevitv2_075_cvnets_in1k": _v(0.75, "mobilevitv2_075.cvnets_in1k"),
    "mobilevitv2_100_cvnets_in1k": _v(1.0, "mobilevitv2_100.cvnets_in1k"),
    "mobilevitv2_125_cvnets_in1k": _v(1.25, "mobilevitv2_125.cvnets_in1k"),
    "mobilevitv2_150_cvnets_in1k": _v(1.5, "mobilevitv2_150.cvnets_in1k"),
    "mobilevitv2_150_cvnets_in22k_ft_in1k": _v(
        1.5, "mobilevitv2_150.cvnets_in22k_ft_in1k"
    ),
    "mobilevitv2_150_cvnets_in22k_ft_in1k_384": _v(
        1.5, "mobilevitv2_150.cvnets_in22k_ft_in1k_384", image_size=384
    ),
    "mobilevitv2_175_cvnets_in1k": _v(1.75, "mobilevitv2_175.cvnets_in1k"),
    "mobilevitv2_175_cvnets_in22k_ft_in1k": _v(
        1.75, "mobilevitv2_175.cvnets_in22k_ft_in1k"
    ),
    "mobilevitv2_175_cvnets_in22k_ft_in1k_384": _v(
        1.75, "mobilevitv2_175.cvnets_in22k_ft_in1k_384", image_size=384
    ),
    "mobilevitv2_200_cvnets_in1k": _v(2.0, "mobilevitv2_200.cvnets_in1k"),
    "mobilevitv2_200_cvnets_in22k_ft_in1k": _v(
        2.0, "mobilevitv2_200.cvnets_in22k_ft_in1k"
    ),
    "mobilevitv2_200_cvnets_in22k_ft_in1k_384": _v(
        2.0, "mobilevitv2_200.cvnets_in22k_ft_in1k_384", image_size=384
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
MOBILEVITV2_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"}
    for variant in MOBILEVITV2_CONFIG
}
