"""ResNeXt variant registry (timm-ported).

Variant ids follow timm naming. ``timm_id`` is the corresponding timm
HF Hub repo, used by both the offline conversion script and the runtime
``ResNeXt.from_weights("timm:...")`` path. ``groups`` / ``width_factor``
are the ResNeXt cardinality knobs.
"""

_50_32X4D = {
    "block_repeats": [3, 4, 6, 3],
    "filters": [64, 128, 256, 512],
    "groups": 32,
    "width_factor": 2,
}
_101_32X4D = {
    "block_repeats": [3, 4, 23, 3],
    "filters": [64, 128, 256, 512],
    "groups": 32,
    "width_factor": 2,
}
_101_32X8D = {
    "block_repeats": [3, 4, 23, 3],
    "filters": [64, 128, 256, 512],
    "groups": 32,
    "width_factor": 4,
}
_101_32X16D = {
    "block_repeats": [3, 4, 23, 3],
    "filters": [64, 128, 256, 512],
    "groups": 32,
    "width_factor": 8,
}
_101_32X32D = {
    "block_repeats": [3, 4, 23, 3],
    "filters": [64, 128, 256, 512],
    "groups": 32,
    "width_factor": 16,
}


def _v(arch, timm_id):
    return {**arch, "timm_id": timm_id}


RESNEXT_CONFIG = {
    "resnext50_32x4d_a1_in1k": _v(_50_32X4D, "resnext50_32x4d.a1_in1k"),
    "resnext50_32x4d_tv_in1k": _v(_50_32X4D, "resnext50_32x4d.tv_in1k"),
    "resnext50_32x4d_gluon_in1k": _v(_50_32X4D, "resnext50_32x4d.gluon_in1k"),
    "resnext101_32x4d_gluon_in1k": _v(_101_32X4D, "resnext101_32x4d.gluon_in1k"),
    "resnext101_32x4d_fb_ssl_yfcc100m_ft_in1k": _v(
        _101_32X4D, "resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k"
    ),
    "resnext101_32x4d_fb_swsl_ig1b_ft_in1k": _v(
        _101_32X4D, "resnext101_32x4d.fb_swsl_ig1b_ft_in1k"
    ),
    "resnext101_32x8d_tv_in1k": _v(_101_32X8D, "resnext101_32x8d.tv_in1k"),
    "resnext101_32x8d_fb_wsl_ig1b_ft_in1k": _v(
        _101_32X8D, "resnext101_32x8d.fb_wsl_ig1b_ft_in1k"
    ),
    "resnext101_32x8d_fb_ssl_yfcc100m_ft_in1k": _v(
        _101_32X8D, "resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k"
    ),
    "resnext101_32x8d_fb_swsl_ig1b_ft_in1k": _v(
        _101_32X8D, "resnext101_32x8d.fb_swsl_ig1b_ft_in1k"
    ),
    "resnext101_32x16d_fb_wsl_ig1b_ft_in1k": _v(
        _101_32X16D, "resnext101_32x16d.fb_wsl_ig1b_ft_in1k"
    ),
    "resnext101_32x16d_fb_ssl_yfcc100m_ft_in1k": _v(
        _101_32X16D, "resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k"
    ),
    "resnext101_32x16d_fb_swsl_ig1b_ft_in1k": _v(
        _101_32X16D, "resnext101_32x16d.fb_swsl_ig1b_ft_in1k"
    ),
    "resnext101_32x32d_fb_wsl_ig1b_ft_in1k": _v(
        _101_32X32D, "resnext101_32x32d.fb_wsl_ig1b_ft_in1k"
    ),
}

_BASE_URL = "https://github.com/IMvision12/keras-models/releases/download/v0.2"
RESNEXT_WEIGHTS = {
    variant: {"url": f"{_BASE_URL}/{variant}.weights.h5"} for variant in RESNEXT_CONFIG
}
