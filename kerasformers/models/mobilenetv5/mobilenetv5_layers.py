import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kerasformers.mobilenetv5")
def gelu_tanh(x):
    """tanh-approximate GELU, matching timm's ``nn.GELU(approximate='tanh')``."""
    return keras.activations.gelu(x, approximate=True)


@keras.saving.register_keras_serializable(package="kerasformers.mobilenetv5")
class RmsNorm2d(layers.Layer):
    """timm ``RmsNorm2d``: RMS normalization over the channel axis (per location).

    ``x * rsqrt(mean(x**2, over channels) + eps) * gamma``, with a length-``C``
    weight and no bias / running statistics (eps 1e-6).
    """

    def __init__(self, epsilon=1e-6, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.data_format = data_format or keras.config.image_data_format()

    def build(self, input_shape):
        self.axis = -1 if self.data_format == "channels_last" else 1
        dim = input_shape[self.axis]
        self.gamma = self.add_weight(
            name="gamma",
            shape=(dim,),
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def call(self, x):
        variance = ops.mean(ops.square(x), axis=self.axis, keepdims=True)
        x = x * ops.rsqrt(variance + self.epsilon)
        if self.axis == -1:
            return x * self.gamma
        return x * ops.reshape(self.gamma, (1, -1, 1, 1))

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon, "data_format": self.data_format})
        return config


@keras.saving.register_keras_serializable(package="kerasformers.mobilenetv5")
class LayerScale2D(layers.Layer):
    """Per-channel learnable scale (timm ``LayerScale2d``): ``x * gamma``."""

    def __init__(self, init_value=1e-5, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value
        self.data_format = data_format or keras.config.image_data_format()

    def build(self, input_shape):
        axis = -1 if self.data_format == "channels_last" else 1
        dim = input_shape[axis]
        self.gamma = self.add_weight(
            name="gamma",
            shape=(dim,),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
        )
        self.built = True

    def call(self, x):
        if self.data_format == "channels_last":
            return x * self.gamma
        return x * ops.reshape(self.gamma, (1, -1, 1, 1))

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value, "data_format": self.data_format})
        return config


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """Snap a (possibly scaled) channel count to a multiple of ``divisor``."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def rms_norm(prefix, data_format, channels_axis, epsilon=1e-6):
    return RmsNorm2d(epsilon=epsilon, data_format=data_format, name=prefix)


def edge_residual(
    x,
    filters,
    exp_kernel_size,
    stride,
    exp_ratio,
    prefix,
    data_format,
    channels_axis,
    norm_epsilon=1e-6,
    noskip=False,
):
    """EdgeResidual / FusedIB block (timm ``er_``) with RmsNorm2d + tanh-GELU."""
    shortcut = x
    in_filters = x.shape[channels_axis]
    mid_filters = make_divisible(in_filters * exp_ratio)

    xp = layers.Conv2D(
        mid_filters,
        kernel_size=exp_kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_conv_exp",
    )(x)
    xp = rms_norm(f"{prefix}_bn1", data_format, channels_axis, norm_epsilon)(xp)
    xp = layers.Activation(gelu_tanh, name=f"{prefix}_act")(xp)

    xp = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_conv_pwl",
    )(xp)
    xp = rms_norm(f"{prefix}_bn2", data_format, channels_axis, norm_epsilon)(xp)

    if not noskip and stride == 1 and in_filters == filters:
        xp = layers.Add(name=f"{prefix}_add")([shortcut, xp])
    return xp


def universal_inverted_residual(
    x,
    filters,
    dw_start_kernel,
    dw_mid_kernel,
    stride,
    exp_ratio,
    prefix,
    data_format,
    channels_axis,
    layer_scale_init=1e-5,
    norm_epsilon=1e-6,
    noskip=False,
):
    """Universal Inverted Bottleneck (timm ``uir_``) with RmsNorm2d + tanh-GELU."""
    shortcut = x
    in_filters = x.shape[channels_axis]
    mid_filters = make_divisible(in_filters * exp_ratio)

    xp = x
    if dw_start_kernel:
        dw_start_stride = stride if not dw_mid_kernel else 1
        xp = layers.DepthwiseConv2D(
            dw_start_kernel,
            strides=dw_start_stride,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=f"{prefix}_dw_start_conv",
        )(xp)
        xp = rms_norm(
            f"{prefix}_dw_start_bn", data_format, channels_axis, norm_epsilon
        )(xp)

    xp = layers.Conv2D(
        mid_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_pw_exp_conv",
    )(xp)
    xp = rms_norm(f"{prefix}_pw_exp_bn", data_format, channels_axis, norm_epsilon)(xp)
    xp = layers.Activation(gelu_tanh, name=f"{prefix}_pw_exp_act")(xp)

    if dw_mid_kernel:
        xp = layers.DepthwiseConv2D(
            dw_mid_kernel,
            strides=stride,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=f"{prefix}_dw_mid_conv",
        )(xp)
        xp = rms_norm(f"{prefix}_dw_mid_bn", data_format, channels_axis, norm_epsilon)(
            xp
        )
        xp = layers.Activation(gelu_tanh, name=f"{prefix}_dw_mid_act")(xp)

    xp = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_pw_proj_conv",
    )(xp)
    xp = rms_norm(f"{prefix}_pw_proj_bn", data_format, channels_axis, norm_epsilon)(xp)

    if layer_scale_init is not None:
        xp = LayerScale2D(
            layer_scale_init, data_format=data_format, name=f"{prefix}_layer_scale"
        )(xp)

    if not noskip and stride == 1 and in_filters == filters:
        xp = layers.Add(name=f"{prefix}_add")([shortcut, xp])
    return xp


def multi_query_attention(
    x,
    dim_out,
    num_heads,
    key_dim,
    value_dim,
    kv_stride,
    dw_kernel_size,
    prefix,
    data_format,
    norm_epsilon=1e-6,
):
    """Mobile Multi-Query Attention 2D (timm ``MultiQueryAttention2d``, RmsNorm2d).

    Single shared key/value head broadcast across ``num_heads`` query heads, with a
    depthwise (stride ``kv_stride``) key/value spatial downsample. TF-style ``same``
    padding throughout. Computed in channels-last layout internally.
    """
    channels_first = data_format == "channels_first"
    if channels_first:
        x = ops.transpose(x, (0, 2, 3, 1))

    height = x.shape[1]
    width = x.shape[2]
    num_tokens = height * width
    scale = key_dim**-0.5

    query = layers.Conv2D(
        num_heads * key_dim,
        kernel_size=1,
        use_bias=False,
        data_format="channels_last",
        name=f"{prefix}_attn_query_proj",
    )(x)
    query = ops.reshape(query, (-1, num_tokens, num_heads, key_dim))
    query = ops.transpose(query, (0, 2, 1, 3))

    key_in = x
    value_in = x
    if kv_stride > 1:
        key_in = layers.DepthwiseConv2D(
            dw_kernel_size,
            strides=kv_stride,
            padding="same",
            use_bias=False,
            data_format="channels_last",
            name=f"{prefix}_attn_key_down_conv",
        )(key_in)
        key_in = RmsNorm2d(
            epsilon=norm_epsilon,
            data_format="channels_last",
            name=f"{prefix}_attn_key_norm",
        )(key_in)
        value_in = layers.DepthwiseConv2D(
            dw_kernel_size,
            strides=kv_stride,
            padding="same",
            use_bias=False,
            data_format="channels_last",
            name=f"{prefix}_attn_value_down_conv",
        )(value_in)
        value_in = RmsNorm2d(
            epsilon=norm_epsilon,
            data_format="channels_last",
            name=f"{prefix}_attn_value_norm",
        )(value_in)

    key = layers.Conv2D(
        key_dim,
        kernel_size=1,
        use_bias=False,
        data_format="channels_last",
        name=f"{prefix}_attn_key_proj",
    )(key_in)
    num_kv = key.shape[1] * key.shape[2]
    key = ops.reshape(key, (-1, 1, num_kv, key_dim))

    value = layers.Conv2D(
        value_dim,
        kernel_size=1,
        use_bias=False,
        data_format="channels_last",
        name=f"{prefix}_attn_value_proj",
    )(value_in)
    value = ops.reshape(value, (-1, 1, num_kv, value_dim))

    attn = ops.matmul(query * scale, ops.transpose(key, (0, 1, 3, 2)))
    attn = ops.softmax(attn, axis=-1)
    out = ops.matmul(attn, value)
    out = ops.transpose(out, (0, 2, 1, 3))
    out = ops.reshape(out, (-1, height, width, num_heads * value_dim))
    out = layers.Conv2D(
        dim_out,
        kernel_size=1,
        use_bias=False,
        data_format="channels_last",
        name=f"{prefix}_attn_output_proj",
    )(out)

    if channels_first:
        out = ops.transpose(out, (0, 3, 1, 2))
    return out


def mobile_attention(
    x,
    dim_out,
    num_heads,
    key_dim,
    value_dim,
    kv_stride,
    dw_kernel_size,
    prefix,
    data_format,
    channels_axis,
    layer_scale_init=1e-5,
    norm_epsilon=1e-6,
    noskip=False,
):
    """MobileAttention block (timm ``mqa_``): RmsNorm -> MQA -> LayerScale -> residual."""
    shortcut = x
    in_filters = x.shape[channels_axis]

    xn = rms_norm(f"{prefix}_norm", data_format, channels_axis, norm_epsilon)(x)
    attn = multi_query_attention(
        xn,
        dim_out=dim_out,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        kv_stride=kv_stride,
        dw_kernel_size=dw_kernel_size,
        prefix=prefix,
        data_format=data_format,
        norm_epsilon=norm_epsilon,
    )

    if layer_scale_init is not None:
        attn = LayerScale2D(
            layer_scale_init, data_format=data_format, name=f"{prefix}_layer_scale"
        )(attn)

    if not noskip and in_filters == dim_out:
        attn = layers.Add(name=f"{prefix}_add")([shortcut, attn])
    return attn


def conv_bn_act(
    x,
    filters,
    kernel_size,
    stride,
    prefix,
    data_format,
    channels_axis,
    norm_epsilon=1e-6,
):
    """ConvBnAct block (timm ``cn_``): conv -> RmsNorm -> tanh-GELU."""
    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_conv",
    )(x)
    x = rms_norm(f"{prefix}_bn1", data_format, channels_axis, norm_epsilon)(x)
    x = layers.Activation(gelu_tanh, name=f"{prefix}_act")(x)
    return x


def multi_scale_fusion_adapter(
    features,
    out_chs,
    output_resolution,
    expansion_ratio,
    prefix,
    data_format,
    channels_axis,
    norm_epsilon=1e-6,
    interpolation="nearest",
):
    """Multi-Scale Fusion Adapter (timm ``MobileNetV5MultiScaleFusionAdapter``).

    Upsamples every lower-resolution feature map to the highest input resolution
    (``features[0]``), channel-concatenates, runs a UIB FFN (pw_exp -> pw_proj, no
    depthwise, no residual, no layer-scale), pools/interpolates to
    ``output_resolution``, then applies a final RmsNorm2d.
    """
    high = (
        features[0].shape[1:3]
        if data_format == "channels_last"
        else (
            features[0].shape[2],
            features[0].shape[3],
        )
    )

    resized = []
    for feat in features:
        size = (
            feat.shape[1:3]
            if data_format == "channels_last"
            else (
                feat.shape[2],
                feat.shape[3],
            )
        )
        if size[0] < high[0] or size[1] < high[1]:
            feat = layers.UpSampling2D(
                size=(high[0] // size[0], high[1] // size[1]),
                data_format=data_format,
                interpolation=interpolation,
            )(feat)
        resized.append(feat)

    x = layers.Concatenate(axis=channels_axis, name=f"{prefix}_concat")(resized)

    in_chs = x.shape[channels_axis]
    mid_chs = make_divisible(in_chs * expansion_ratio)
    x = layers.Conv2D(
        mid_chs,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_ffn_pw_exp_conv",
    )(x)
    x = rms_norm(f"{prefix}_ffn_pw_exp_bn", data_format, channels_axis, norm_epsilon)(x)
    x = layers.Activation(gelu_tanh, name=f"{prefix}_ffn_pw_exp_act")(x)
    x = layers.Conv2D(
        out_chs,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_ffn_pw_proj_conv",
    )(x)
    x = rms_norm(f"{prefix}_ffn_pw_proj_bn", data_format, channels_axis, norm_epsilon)(
        x
    )

    if high[0] != output_resolution or high[1] != output_resolution:
        if high[0] % output_resolution == 0 and high[1] % output_resolution == 0:
            x = layers.AveragePooling2D(
                pool_size=(high[0] // output_resolution, high[1] // output_resolution),
                strides=(high[0] // output_resolution, high[1] // output_resolution),
                padding="valid",
                data_format=data_format,
                name=f"{prefix}_pool",
            )(x)
        else:
            x = layers.Resizing(
                output_resolution,
                output_resolution,
                interpolation="bilinear",
                data_format=data_format,
                name=f"{prefix}_resize",
            )(x)

    x = rms_norm(f"{prefix}_norm", data_format, channels_axis, norm_epsilon)(x)
    return x


def decode_block_str(block_str):
    """Parse a timm arch-def block string into ``(block_type, options, noskip)``."""
    parts = block_str.split("_")
    block_type = parts[0]
    options = {}
    noskip = False
    for token in parts[1:]:
        if token == "noskip":
            noskip = True
        elif token == "skip":
            noskip = False
        else:
            options[token[0]] = token[1:]
    return block_type, options, noskip
