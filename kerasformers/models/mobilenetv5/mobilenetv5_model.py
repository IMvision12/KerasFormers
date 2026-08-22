import keras
from keras import layers, utils

from kerasformers.base import FunctionalBaseModel
from kerasformers.utils import standardize_input_shape
from kerasformers.utils.image_util import normalize_image_for_classify_models

from .mobilenetv5_config import MobileNetV5Config
from .mobilenetv5_layers import (
    RmsNorm2d,
    conv_bn_act,
    decode_block_str,
    edge_residual,
    gelu_tanh,
    make_divisible,
    mobile_attention,
    multi_scale_fusion_adapter,
    universal_inverted_residual,
)

MOBILENETV5_HUB_SIBLINGS = frozenset({"MobileNetV5Encoder"})

# MSFA fused output width (head_hidden_size), shared across variants.
MSFA_OUT_CHANNELS = 2048
MSFA_EXPANSION_RATIO = 2.0

# The 300m arch-def, transcribed from timm's ``_gen_mobilenet_v5`` (else branch,
# used by both ``mobilenetv5_300m`` and ``mobilenetv5_300m_enc``). Built from
# repeat helpers to mirror the timm ``rN`` block-string repeats exactly.
_ARCH_300M = [
    # Stage 0 (EdgeResidual, out 128)
    [
        "er_r1_k3_s2_e4_c128",
        "er_r1_k3_s1_e4_c128",
        "er_r1_k3_s1_e4_c128",
    ],
    # Stage 1 (UIB, out 256)
    [
        "uir_r1_a3_k5_s2_e6_c256",
        "uir_r1_a5_k0_s1_e4_c256",
        "uir_r1_a3_k0_s1_e4_c256",
        "uir_r1_a5_k0_s1_e4_c256",
        "uir_r1_a3_k0_s1_e4_c256",
    ],
    # Stage 2 (UIB + Mobile MQA, out 640): downsample + 7 ExtraDW + 1 IB + 14 (MQA, FFN)
    (
        ["uir_r1_a5_k5_s2_e6_c640"]
        + ["uir_r1_a5_k0_s1_e4_c640"] * 7
        + ["uir_r1_a0_k0_s1_e1_c640"]
        + ["mqa_r1_k3_h12_v2_s1_d64_c640", "uir_r1_a0_k0_s1_e2_c640"] * 14
    ),
    # Stage 3 (UIB + Mobile MQA, out 1280): downsample + 19 (MQA, FFN)
    (
        ["uir_r1_a5_k5_s2_e6_c1280"]
        + ["mqa_r1_k3_h16_s1_d96_c1280", "uir_r1_a0_k0_s1_e2_c1280"] * 19
    ),
]

MOBILENETV5_VARIANTS = {
    "300m": {"stem_size": 64, "arch": _ARCH_300M},
}


def build_block(
    x,
    block_type,
    options,
    noskip,
    stride,
    layer_scale_init,
    prefix,
    data_format,
    channels_axis,
    norm_epsilon,
):
    """Dispatch a decoded arch-def block onto its Keras block builder."""
    out_chs = make_divisible(int(options["c"]))
    if block_type == "cn":
        return conv_bn_act(
            x,
            filters=out_chs,
            kernel_size=int(options["k"]),
            stride=stride,
            prefix=prefix,
            data_format=data_format,
            channels_axis=channels_axis,
            norm_epsilon=norm_epsilon,
        )
    if block_type == "er":
        return edge_residual(
            x,
            filters=out_chs,
            exp_kernel_size=int(options["k"]),
            stride=stride,
            exp_ratio=float(options["e"]),
            prefix=prefix,
            data_format=data_format,
            channels_axis=channels_axis,
            norm_epsilon=norm_epsilon,
            noskip=noskip,
        )
    if block_type == "uir":
        return universal_inverted_residual(
            x,
            filters=out_chs,
            dw_start_kernel=int(options["a"]),
            dw_mid_kernel=int(options["k"]),
            stride=stride,
            exp_ratio=float(options["e"]),
            prefix=prefix,
            data_format=data_format,
            channels_axis=channels_axis,
            layer_scale_init=layer_scale_init,
            norm_epsilon=norm_epsilon,
            noskip=noskip,
        )
    if block_type == "mqa":
        key_dim = int(options["d"])
        return mobile_attention(
            x,
            dim_out=out_chs,
            num_heads=int(options["h"]),
            key_dim=key_dim,
            value_dim=key_dim,
            kv_stride=int(options.get("v", 1)),
            dw_kernel_size=int(options["k"]),
            prefix=prefix,
            data_format=data_format,
            channels_axis=channels_axis,
            layer_scale_init=layer_scale_init,
            norm_epsilon=norm_epsilon,
            noskip=noskip,
        )
    raise ValueError(f"Unknown MobileNetV5 block type: {block_type!r}")


def mobilenetv5_encoder_feature(
    inputs,
    *,
    arch,
    stem_size,
    msfa_output_resolution,
    data_format,
    channels_axis,
    layer_scale_init=1e-5,
    norm_epsilon=1e-6,
):
    """MobileNetV5 stem + arch-def stages + Multi-Scale Fusion Adapter.

    Returns the MSFA-fused feature map of shape ``(B, R, R, 2048)`` where ``R`` is
    ``msfa_output_resolution``. The MSFA consumes the last two stage outputs.
    """
    stem_size = make_divisible(stem_size)

    x = layers.Conv2D(
        stem_size,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=True,
        data_format=data_format,
        name="conv_stem_conv",
    )(inputs)
    x = RmsNorm2d(epsilon=norm_epsilon, data_format=data_format, name="conv_stem_bn")(x)
    x = layers.Activation(gelu_tanh, name="conv_stem_act")(x)

    stage_outputs = []
    for stage_idx, stage in enumerate(arch):
        block_idx = 0
        for block_str in stage:
            block_type, options, noskip = decode_block_str(block_str)
            repeat = int(options.get("r", 1))
            for rep in range(repeat):
                stride = int(options["s"]) if rep == 0 else 1
                x = build_block(
                    x,
                    block_type=block_type,
                    options=options,
                    noskip=noskip,
                    stride=stride,
                    layer_scale_init=layer_scale_init,
                    prefix=f"blocks_{stage_idx}_{block_idx}",
                    data_format=data_format,
                    channels_axis=channels_axis,
                    norm_epsilon=norm_epsilon,
                )
                block_idx += 1
        stage_outputs.append(x)

    # MSFA consumes the last two stage outputs (highest-res first).
    msfa_inputs = [stage_outputs[-2], stage_outputs[-1]]
    x = multi_scale_fusion_adapter(
        msfa_inputs,
        out_chs=MSFA_OUT_CHANNELS,
        output_resolution=msfa_output_resolution,
        expansion_ratio=MSFA_EXPANSION_RATIO,
        prefix="msfa",
        data_format=data_format,
        channels_axis=channels_axis,
        norm_epsilon=norm_epsilon,
    )
    return x


@keras.saving.register_keras_serializable(package="kerasformers.mobilenetv5")
class MobileNetV5Encoder(FunctionalBaseModel):
    """Instantiates the MobileNetV5 vision encoder.

    MobileNetV5 is the vision tower introduced with Gemma 3n. It reuses the
    MobileNetV4 block vocabulary (EdgeResidual, Universal Inverted Bottleneck, and
    Mobile Multi-Query Attention) but replaces BatchNorm with RmsNorm2d (channel-wise
    RMS normalization), uses tanh-approximate GELU, TF-style ``same`` padding, and a
    biased ConvNormAct stem. Instead of a classifier head it ends in a Multi-Scale
    Fusion Adapter (MSFA): the last two stage feature maps are upsampled to a common
    resolution, channel-concatenated, passed through an inverted-bottleneck FFN, then
    pooled to an ``R x R`` grid and RMS-normalized, yielding ``R * R`` tokens of width
    2048.

    Output is the MSFA feature map of shape ``(B, R, R, 2048)`` (``R`` =
    ``msfa_output_resolution``, 16 by default). This is a standalone port of timm's
    ``mobilenetv5_300m_enc``; it takes no code from the Gemma 3n vision tower.

    References:
    - [Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n)

    Args:
        config: String, variant key selecting the block schedule. Currently
            ``"300m"``. Defaults to `"300m"`.
        msfa_output_resolution: Integer, spatial size of the MSFA output grid.
            Defaults to `16`.
        norm_epsilon: Float, epsilon for every RmsNorm2d layer. Defaults to `1e-6`.
        layer_scale_init: Float, LayerScale init value for UIB / attention blocks.
            Defaults to `1e-5`.
        image_size: Input image specification. Accepts an integer ``N`` (builds an
            ``N x N x 3`` square input), a 2-tuple ``(H, W)``, or a 3-tuple ordered
            to match ``keras.config.image_data_format()``. Defaults to `768`.
        include_normalization: Boolean, whether to prepend image normalization.
            The Gemma 3n encoder expects externally normalized input, so this
            defaults to `False`.
        normalization_mode: String, normalization mode when
            ``include_normalization=True``. Defaults to `"minus_one_to_one"`.
        input_tensor: Optional Keras tensor as input. Defaults to `None`.
        name: String, the name of the model. Defaults to `"MobileNetV5Encoder"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = MobileNetV5Config
    HUB_REPO_SIBLINGS = MOBILENETV5_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_mobilenetv5_timm_to_keras import transfer_mobilenetv5_weights

        transfer_mobilenetv5_weights(keras_model, state_dict)

    def __init__(
        self,
        config="300m",
        arch=None,
        stem_size=None,
        msfa_output_resolution=16,
        norm_epsilon=1e-6,
        layer_scale_init=1e-5,
        image_size=768,
        include_normalization=False,
        normalization_mode="minus_one_to_one",
        input_tensor=None,
        name="MobileNetV5Encoder",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        if config not in MOBILENETV5_VARIANTS:
            raise ValueError(
                f"Invalid config {config!r}. Expected one of "
                f"{sorted(MOBILENETV5_VARIANTS)}"
            )

        # ``arch`` / ``stem_size`` override the variant's schedule (used to build
        # smaller custom encoders); default to the selected variant otherwise.
        spec = MOBILENETV5_VARIANTS[config]
        resolved_arch = arch if arch is not None else spec["arch"]
        resolved_stem = stem_size if stem_size is not None else spec["stem_size"]

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=image_size)
        else:
            img_input = input_tensor

        x = (
            normalize_image_for_classify_models(img_input, normalization_mode)
            if include_normalization
            else img_input
        )
        x = mobilenetv5_encoder_feature(
            x,
            arch=resolved_arch,
            stem_size=resolved_stem,
            msfa_output_resolution=msfa_output_resolution,
            data_format=data_format,
            channels_axis=channels_axis,
            layer_scale_init=layer_scale_init,
            norm_epsilon=norm_epsilon,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.config_name = config
        self.arch = arch
        self.stem_size = stem_size
        self.msfa_output_resolution = msfa_output_resolution
        self.norm_epsilon = norm_epsilon
        self.layer_scale_init = layer_scale_init
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.config_name,
                "arch": self.arch,
                "stem_size": self.stem_size,
                "msfa_output_resolution": self.msfa_output_resolution,
                "norm_epsilon": self.norm_epsilon,
                "layer_scale_init": self.layer_scale_init,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
