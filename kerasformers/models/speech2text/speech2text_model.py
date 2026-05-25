from typing import List, Union

import keras
import numpy as np
from keras import layers, ops

from kerasformers.base import BaseModel

from .config import SPEECH2TEXT_CONFIG, SPEECH2TEXT_WEIGHTS
from .speech2text_layers import (
    Speech2TextAttention,
    Speech2TextSinusoidalPositionEmbedding,
)

_ACTIVATION_ALIASES = {
    "relu": keras.activations.relu,
    "gelu": lambda x: keras.activations.gelu(x, approximate=False),
    "gelu_new": lambda x: keras.activations.gelu(x, approximate=True),
    "silu": keras.activations.silu,
    "swish": keras.activations.silu,
}


def _relu(x):
    """Module-level ReLU so it round-trips through ``Lambda`` serialization."""
    return keras.activations.relu(x)


def _resolve_activation(name):
    """Return a callable for the activation name (Speech2Text uses ``relu``)."""
    if callable(name):
        return name
    if name == "relu":
        return _relu
    if name in _ACTIVATION_ALIASES:
        return _ACTIVATION_ALIASES[name]
    return keras.activations.get(name)


def _glu(x):
    """Gated Linear Unit over the channel (last) axis: ``a * sigmoid(b)``."""
    a, b = ops.split(x, 2, axis=-1)
    return a * ops.sigmoid(b)


def speech2text_conv_subsampler(
    features,
    hidden_dim,
    conv_channels,
    conv_kernel_sizes,
    num_conv_layers,
    name="encoder",
):
    """Conv1d feature subsampler (the Speech2Text encoder front-end).

    Stacks ``num_conv_layers`` strided ``Conv1D`` layers, each followed by a
    GLU that halves the channel dimension. With two kernel-5 / stride-2 layers
    the time axis is downsampled by 4x while the channel dimension goes
    ``num_mel_bins -> conv_channels/2 -> hidden_dim``. Explicit
    ``ZeroPadding1D(k // 2)`` reproduces the reference's symmetric Conv1d
    padding so output lengths and frame alignment match exactly.

    Args:
        features: Input fbank features ``(B, T, num_mel_bins)``.
        hidden_dim: Encoder model dimension (the final output channels).
        conv_channels: Intermediate channel width before each GLU halving.
        conv_kernel_sizes: Per-layer kernel sizes (e.g. ``(5, 5)``).
        num_conv_layers: Number of conv+GLU layers.
        name: Prefix used for the conv layer names so the source state-dict
            (``encoder.conv.conv_layers.{i}``) transfers by name.

    Returns:
        Subsampled tensor ``(B, T // 2**num_conv_layers, hidden_dim)``.
    """
    x = features
    for i in range(num_conv_layers):
        out_ch = conv_channels if i < num_conv_layers - 1 else hidden_dim * 2
        k = conv_kernel_sizes[i]
        x = layers.ZeroPadding1D(padding=k // 2, name=f"{name}_conv_layers_{i}_pad")(x)
        x = layers.Conv1D(
            filters=out_ch,
            kernel_size=k,
            strides=2,
            padding="valid",
            name=f"{name}_conv_layers_{i}",
        )(x)
        x = layers.Lambda(_glu, name=f"{name}_conv_layers_{i}_glu")(x)
    return x


def speech2text_encoder_block(
    x, hidden_dim, num_heads, mlp_dim, layer_idx, activation, layer_norm_eps
):
    """One pre-LN Speech2Text encoder block (self-attention + MLP).

    Pre-norm: each sublayer normalizes its input, then the result is added back
    to the residual (``x = x + sublayer(LN(x))``), matching the reference. A
    single ``encoder_layer_norm`` is applied once after the whole stack.
    Sublayer names follow ``encoder_layers_{layer_idx}_*`` for by-name transfer.
    """
    prefix = f"encoder_layers_{layer_idx}"

    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_self_attn_layer_norm"
    )(x)
    h = Speech2TextAttention(
        proj_dim=hidden_dim, num_heads=num_heads, name_prefix=f"{prefix}_self_attn"
    )(h)
    x = layers.Add()([residual, h])

    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_final_layer_norm"
    )(x)
    h = layers.Dense(mlp_dim, name=f"{prefix}_fc1")(h)
    h = layers.Lambda(activation, name=f"{prefix}_fc1_act")(h)
    h = layers.Dense(hidden_dim, name=f"{prefix}_fc2")(h)
    x = layers.Add()([residual, h])
    return x


def speech2text_decoder_block(
    x,
    encoder_hidden_states,
    causal_mask,
    hidden_dim,
    num_heads,
    mlp_dim,
    layer_idx,
    activation,
    layer_norm_eps,
):
    """One pre-LN Speech2Text decoder block (causal self-attn + cross-attn + MLP).

    Three pre-norm sublayers - causal self-attention, cross-attention over the
    encoder output, and an MLP - each ``x = x + sublayer(LN(x))``. Sublayer
    names follow ``decoder_layers_{layer_idx}_*``.
    """
    prefix = f"decoder_layers_{layer_idx}"

    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_self_attn_layer_norm"
    )(x)
    h = Speech2TextAttention(
        proj_dim=hidden_dim, num_heads=num_heads, name_prefix=f"{prefix}_self_attn"
    )(h, attention_mask=causal_mask)
    x = layers.Add()([residual, h])

    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_encoder_attn_layer_norm"
    )(x)
    h = Speech2TextAttention(
        proj_dim=hidden_dim, num_heads=num_heads, name_prefix=f"{prefix}_encoder_attn"
    )(h, key_value_states=encoder_hidden_states)
    x = layers.Add()([residual, h])

    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_final_layer_norm"
    )(x)
    h = layers.Dense(mlp_dim, name=f"{prefix}_fc1")(h)
    h = layers.Lambda(activation, name=f"{prefix}_fc1_act")(h)
    h = layers.Dense(hidden_dim, name=f"{prefix}_fc2")(h)
    x = layers.Add()([residual, h])
    return x


def _make_causal_mask_from_ids(decoder_input_ids):
    """Additive ``(1, 1, L, L)`` causal mask sized to the decoder length."""
    seq_len = ops.shape(decoder_input_ids)[1]
    i = ops.arange(seq_len)[:, None]
    j = ops.arange(seq_len)[None, :]
    mask = ops.cast(j > i, "float32") * -1e9
    return mask[None, None, :, :]


def speech2text_encoder(
    hidden_dim,
    num_mel_bins,
    conv_channels,
    conv_kernel_sizes,
    num_conv_layers,
    max_source_positions,
    encoder_num_layers,
    encoder_attention_heads,
    encoder_ffn_dim,
    embed_scale,
    activation,
    layer_norm_eps,
    name="encoder",
):
    """Build the Speech2Text encoder (conv subsampler + post-LN transformer)."""
    features = layers.Input(shape=(None, num_mel_bins), name="input_features")
    x = speech2text_conv_subsampler(
        features,
        hidden_dim,
        conv_channels,
        conv_kernel_sizes,
        num_conv_layers,
        name="encoder",
    )
    if embed_scale != 1.0:
        x = layers.Lambda(lambda t, s=embed_scale: t * s, name="encoder_embed_scale")(x)
    x = Speech2TextSinusoidalPositionEmbedding(
        num_positions=max_source_positions,
        hidden_dim=hidden_dim,
        name="encoder_embed_positions",
    )(x)
    for i in range(encoder_num_layers):
        x = speech2text_encoder_block(
            x,
            hidden_dim=hidden_dim,
            num_heads=encoder_attention_heads,
            mlp_dim=encoder_ffn_dim,
            layer_idx=i,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )
    x = layers.LayerNormalization(epsilon=layer_norm_eps, name="encoder_layer_norm")(x)
    return keras.Model(inputs=features, outputs=x, name=name)


def speech2text_decoder(
    hidden_dim,
    max_target_positions,
    vocab_size,
    decoder_num_layers,
    decoder_attention_heads,
    decoder_ffn_dim,
    embed_scale,
    activation,
    layer_norm_eps,
    pad_token_id,
    name="decoder",
):
    """Build the Speech2Text decoder (token + sinusoidal embeds + post-LN + lm_head)."""
    decoder_input_ids = layers.Input(
        shape=(None,), dtype="int32", name="decoder_input_ids"
    )
    encoder_hidden_states = layers.Input(
        shape=(None, hidden_dim), name="encoder_hidden_states"
    )

    tok_embed = layers.Embedding(
        input_dim=vocab_size, output_dim=hidden_dim, name="decoder_embed_tokens"
    )
    x = tok_embed(decoder_input_ids)
    if embed_scale != 1.0:
        x = layers.Lambda(lambda t, s=embed_scale: t * s, name="decoder_embed_scale")(x)
    x = Speech2TextSinusoidalPositionEmbedding(
        num_positions=max_target_positions,
        hidden_dim=hidden_dim,
        padding_idx=pad_token_id,
        name="decoder_embed_positions",
    )(x)

    causal_mask = layers.Lambda(
        _make_causal_mask_from_ids,
        name="decoder_causal_mask",
        output_shape=lambda s: (1, 1, s[1], s[1]),
    )(decoder_input_ids)

    for i in range(decoder_num_layers):
        x = speech2text_decoder_block(
            x,
            encoder_hidden_states=encoder_hidden_states,
            causal_mask=causal_mask,
            hidden_dim=hidden_dim,
            num_heads=decoder_attention_heads,
            mlp_dim=decoder_ffn_dim,
            layer_idx=i,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )

    x = layers.LayerNormalization(epsilon=layer_norm_eps, name="decoder_layer_norm")(x)
    logits = layers.Dense(vocab_size, use_bias=False, name="lm_head")(x)

    return keras.Model(
        inputs={
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        },
        outputs=logits,
        name=name,
    )


@keras.saving.register_keras_serializable(package="kerasformers")
class Speech2TextModel(BaseModel):
    """Speech2Text (S2T) conv-Transformer encoder-decoder for ASR / ST.

    Wires :func:`speech2text_encoder` and :func:`speech2text_decoder`
    into one Functional graph callable with a single dict:

    >>> out = model({"input_features": fbank, "decoder_input_ids": ids})
    >>> out["encoder_hidden_states"]   # (B, T // 4, hidden_dim)
    >>> out["logits"]                  # (B, L, vocab_size)

    This is the teacher-forced path. For autoregressive transcription use
    :class:`Speech2TextSpeechToText`, which adds ``.generate(audio, processor)``.

    Construction:

    >>> Speech2TextModel.from_weights("s2t-small-librispeech-asr")
    >>> Speech2TextModel.from_weights("hf:facebook/s2t-small-librispeech-asr")

    .. note::
        Like Whisper, the audio input shape is dictated by the feature
        pipeline: 80-channel fbank features ``(num_mel_bins, T)`` fed as
        ``(B, T, num_mel_bins)``. There is no ``input_shape`` kwarg; feed
        :class:`Speech2TextFeatureExtractor` output directly.

    Args:
        hidden_dim: Hidden / embedding dimension (``d_model``).
        encoder_num_layers: Number of encoder transformer blocks.
        decoder_num_layers: Number of decoder transformer blocks.
        encoder_attention_heads: Encoder self-attn head count.
        decoder_attention_heads: Decoder self-/cross-attn head count.
        encoder_ffn_dim: Encoder MLP hidden dim.
        decoder_ffn_dim: Decoder MLP hidden dim.
        vocab_size: Token vocabulary size.
        num_mel_bins: Input fbank feature dimension (``input_feat_per_channel``).
        max_source_positions: Max encoder position (sinusoid table size).
        max_target_positions: Max decoder position.
        conv_channels: Intermediate channel width in the conv subsampler.
        conv_kernel_sizes: Per-layer conv kernel sizes.
        num_conv_layers: Number of conv+GLU subsampling layers.
        scale_embedding: Multiply embeddings by ``sqrt(hidden_dim)``.
        activation_function: FFN activation. ``"relu"`` for S2T.
        layer_norm_eps: Epsilon for every LayerNorm.
        pad_token_id: Padding id (offsets the sinusoidal positions).
        name: Model name.
    """

    BASE_MODEL_CONFIG = SPEECH2TEXT_CONFIG
    BASE_WEIGHT_CONFIG = SPEECH2TEXT_WEIGHTS
    HF_MODEL_TYPE = "speech_to_text"

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "hidden_dim": hf_config["d_model"],
            "encoder_num_layers": hf_config["encoder_layers"],
            "decoder_num_layers": hf_config["decoder_layers"],
            "encoder_attention_heads": hf_config["encoder_attention_heads"],
            "decoder_attention_heads": hf_config["decoder_attention_heads"],
            "encoder_ffn_dim": hf_config["encoder_ffn_dim"],
            "decoder_ffn_dim": hf_config["decoder_ffn_dim"],
            "vocab_size": hf_config["vocab_size"],
            "num_mel_bins": hf_config.get("input_feat_per_channel", 80),
            "max_source_positions": hf_config.get("max_source_positions", 6000),
            "max_target_positions": hf_config.get("max_target_positions", 1024),
            "conv_channels": hf_config.get("conv_channels", 1024),
            "conv_kernel_sizes": tuple(hf_config.get("conv_kernel_sizes", [5, 5])),
            "num_conv_layers": hf_config.get("num_conv_layers", 2),
            "scale_embedding": hf_config.get("scale_embedding", True),
            "activation_function": hf_config.get("activation_function", "relu"),
            "pad_token_id": hf_config.get("pad_token_id", 1),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_speech2text_hf_to_keras import transfer_speech2text_weights

        transfer_speech2text_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        hidden_dim=256,
        encoder_num_layers=12,
        decoder_num_layers=6,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        vocab_size=10000,
        num_mel_bins=80,
        max_source_positions=6000,
        max_target_positions=1024,
        conv_channels=1024,
        conv_kernel_sizes=(5, 5),
        num_conv_layers=2,
        scale_embedding=True,
        activation_function="relu",
        layer_norm_eps=1e-5,
        pad_token_id=1,
        name="Speech2TextModel",
        **kwargs,
    ):
        conv_kernel_sizes = tuple(conv_kernel_sizes)
        activation_fn = _resolve_activation(activation_function)
        embed_scale = float(hidden_dim) ** 0.5 if scale_embedding else 1.0

        encoder = speech2text_encoder(
            hidden_dim=hidden_dim,
            num_mel_bins=num_mel_bins,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            num_conv_layers=num_conv_layers,
            max_source_positions=max_source_positions,
            encoder_num_layers=encoder_num_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            embed_scale=embed_scale,
            activation=activation_fn,
            layer_norm_eps=layer_norm_eps,
            name=f"{name}_encoder",
        )
        decoder = speech2text_decoder(
            hidden_dim=hidden_dim,
            max_target_positions=max_target_positions,
            vocab_size=vocab_size,
            decoder_num_layers=decoder_num_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            embed_scale=embed_scale,
            activation=activation_fn,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            name=f"{name}_decoder",
        )

        input_features = layers.Input(shape=(None, num_mel_bins), name="input_features")
        decoder_input_ids = layers.Input(
            shape=(None,), dtype="int32", name="decoder_input_ids"
        )
        encoder_hidden_states = encoder(input_features)
        logits = decoder(
            {
                "decoder_input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states,
            }
        )

        super().__init__(
            inputs={
                "input_features": input_features,
                "decoder_input_ids": decoder_input_ids,
            },
            outputs={
                "encoder_hidden_states": encoder_hidden_states,
                "logits": logits,
            },
            name=name,
            **kwargs,
        )

        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dim = hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.num_conv_layers = num_conv_layers
        self.scale_embedding = scale_embedding
        self.activation_function = activation_function
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "encoder_num_layers": self.encoder_num_layers,
                "decoder_num_layers": self.decoder_num_layers,
                "encoder_attention_heads": self.encoder_attention_heads,
                "decoder_attention_heads": self.decoder_attention_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "vocab_size": self.vocab_size,
                "num_mel_bins": self.num_mel_bins,
                "max_source_positions": self.max_source_positions,
                "max_target_positions": self.max_target_positions,
                "conv_channels": self.conv_channels,
                "conv_kernel_sizes": self.conv_kernel_sizes,
                "num_conv_layers": self.num_conv_layers,
                "scale_embedding": self.scale_embedding,
                "activation_function": self.activation_function,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class Speech2TextSpeechToText(Speech2TextModel):
    """Speech2Text speech-to-text model (transcription / speech translation).

    Composes the same encoder + decoder + ``lm_head`` Functional graph as
    :class:`Speech2TextModel` (loads the same weights), and adds
    :meth:`generate` - an end-to-end audio -> text method that runs the
    feature extractor, encoder, autoregressive greedy decoding, and
    detokenization via a :class:`Speech2TextProcessor`.

    .. code-block:: python

        model = Speech2TextSpeechToText.from_weights("s2t-small-librispeech-asr")
        processor = Speech2TextProcessor.from_weights("s2t-small-librispeech-asr")
        text = model.generate(audio, processor)
    """

    def generate(
        self,
        audio,
        processor,
        max_new_tokens: int = 200,
        sampling_rate: int = 16000,
        return_ids: bool = False,
    ) -> Union[List[str], List[List[int]]]:
        """End-to-end audio -> text using a :class:`Speech2TextProcessor`.

        Runs fbank extraction, the encoder, greedy autoregressive decoding
        seeded with ``decoder_start_token_id`` (``</s>``), and detokenization.

        Args:
            audio: 1-D waveform or a list / batch of waveforms at
                ``sampling_rate`` Hz.
            processor: A :class:`Speech2TextProcessor`.
            max_new_tokens: Maximum decoded tokens.
            sampling_rate: Input sampling rate (must be 16000).
            return_ids: When ``True``, return raw token-id lists instead of
                decoded strings.
        """
        inputs = processor(audio=audio, sampling_rate=sampling_rate)
        decoder_start_token_id = processor.decoder_start_token_id
        eos_token_id = processor.tokenizer.eos_token_id

        enc_out = self.encoder(inputs["input_features"])
        enc_np = (
            ops.convert_to_numpy(enc_out)
            if not isinstance(enc_out, np.ndarray)
            else enc_out
        )
        batch = enc_np.shape[0]

        generated = np.full((batch, 1), decoder_start_token_id, dtype=np.int32)
        done = np.zeros(batch, dtype=bool)

        for _ in range(max_new_tokens):
            logits = self.decoder(
                {"decoder_input_ids": generated, "encoder_hidden_states": enc_np}
            )
            next_ids = np.argmax(
                ops.convert_to_numpy(logits)[:, -1, :], axis=-1
            ).astype(np.int32)
            next_ids = np.where(done, eos_token_id, next_ids)
            generated = np.concatenate([generated, next_ids[:, None]], axis=1)
            done = done | (next_ids == eos_token_id)
            if done.all():
                break

        ids = [list(row) for row in generated]
        if return_ids:
            return ids
        return processor.batch_decode(ids, skip_special_tokens=True)
