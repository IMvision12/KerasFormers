from typing import List, Optional, Union

import keras
from keras import layers, ops

from kerasformers.base import FunctionalBaseModel

from .config import (
    WHISPER_BEGIN_SUPPRESS_TOKENS,
    WHISPER_CONFIG,
    WHISPER_SUPPRESS_TOKENS,
    WHISPER_WEIGHTS,
)
from .whisper_layers import (
    WhisperAttention,
    WhisperLayerWeights,
    WhisperLearnedPositionEmbedding,
    WhisperSinusoidalPositionEmbedding,
)

_ACTIVATION_ALIASES = {
    "gelu": lambda x: keras.activations.gelu(x, approximate=False),
    "gelu_new": lambda x: keras.activations.gelu(x, approximate=True),
    "relu": keras.activations.relu,
    "silu": keras.activations.silu,
    "swish": keras.activations.silu,
}


def _gelu(x):
    return keras.activations.gelu(x, approximate=False)


def _resolve_activation(name):
    if callable(name):
        return name
    if name in _ACTIVATION_ALIASES:
        return _ACTIVATION_ALIASES[name]
    return keras.activations.get(name)


def whisper_encoder_block(
    x,
    hidden_dim,
    num_heads,
    mlp_dim,
    layer_idx,
    activation=_gelu,
    layer_norm_eps=1e-5,
):
    prefix = f"encoder_layers_{layer_idx}"

    ln_1 = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_self_attn_layer_norm"
    )(x)
    attn_out = WhisperAttention(
        proj_dim=hidden_dim,
        num_heads=num_heads,
        name_prefix=f"{prefix}_self_attn",
    )(ln_1)
    x = layers.Add()([x, attn_out])

    ln_2 = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_final_layer_norm"
    )(x)
    h = layers.Dense(mlp_dim, name=f"{prefix}_fc1")(ln_2)
    h = layers.Lambda(activation, name=f"{prefix}_fc1_act")(h)
    h = layers.Dense(hidden_dim, name=f"{prefix}_fc2")(h)
    x = layers.Add()([x, h])
    return x


def whisper_decoder_block(
    x,
    encoder_hidden_states,
    causal_mask,
    hidden_dim,
    num_heads,
    mlp_dim,
    layer_idx,
    activation=_gelu,
    layer_norm_eps=1e-5,
):
    prefix = f"decoder_layers_{layer_idx}"

    ln_1 = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_self_attn_layer_norm"
    )(x)
    self_attn_out = WhisperAttention(
        proj_dim=hidden_dim,
        num_heads=num_heads,
        name_prefix=f"{prefix}_self_attn",
    )(ln_1, attention_mask=causal_mask)
    x = layers.Add()([x, self_attn_out])

    ln_2 = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_encoder_attn_layer_norm"
    )(x)
    cross_attn_out = WhisperAttention(
        proj_dim=hidden_dim,
        num_heads=num_heads,
        name_prefix=f"{prefix}_encoder_attn",
    )(ln_2, key_value_states=encoder_hidden_states)
    x = layers.Add()([x, cross_attn_out])

    ln_3 = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_final_layer_norm"
    )(x)
    h = layers.Dense(mlp_dim, name=f"{prefix}_fc1")(ln_3)
    h = layers.Lambda(activation, name=f"{prefix}_fc1_act")(h)
    h = layers.Dense(hidden_dim, name=f"{prefix}_fc2")(h)
    x = layers.Add()([x, h])
    return x


def whisper_encoder(
    hidden_dim,
    num_mel_bins,
    max_source_positions,
    encoder_num_layers,
    encoder_attention_heads,
    encoder_ffn_dim,
    activation=_gelu,
    layer_norm_eps=1e-5,
    output_all_hidden_states=False,
    name="encoder",
):
    mel = layers.Input(shape=(num_mel_bins, None), name="input_features")
    x = layers.Permute((2, 1), name="encoder_permute_in")(mel)

    x = layers.ZeroPadding1D(padding=1, name="encoder_conv1_pad")(x)
    x = layers.Conv1D(
        filters=hidden_dim,
        kernel_size=3,
        strides=1,
        padding="valid",
        name="encoder_conv1",
    )(x)
    x = layers.Lambda(activation, name="encoder_conv1_act")(x)
    x = layers.ZeroPadding1D(padding=1, name="encoder_conv2_pad")(x)
    x = layers.Conv1D(
        filters=hidden_dim,
        kernel_size=3,
        strides=2,
        padding="valid",
        name="encoder_conv2",
    )(x)
    x = layers.Lambda(activation, name="encoder_conv2_act")(x)

    x = WhisperSinusoidalPositionEmbedding(
        max_source_positions=max_source_positions,
        hidden_dim=hidden_dim,
        name="encoder_embed_positions",
    )(x)

    all_hidden = [x]
    for i in range(encoder_num_layers):
        x = whisper_encoder_block(
            x,
            hidden_dim=hidden_dim,
            num_heads=encoder_attention_heads,
            mlp_dim=encoder_ffn_dim,
            layer_idx=i,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )
        all_hidden.append(x)

    x = layers.LayerNormalization(epsilon=layer_norm_eps, name="encoder_layer_norm")(x)

    if output_all_hidden_states:
        all_hidden[-1] = x
        return keras.Model(inputs=mel, outputs=all_hidden, name=name)
    return keras.Model(inputs=mel, outputs=x, name=name)


def make_causal_mask_from_ids(decoder_input_ids):
    seq_len = ops.shape(decoder_input_ids)[1]
    i = ops.arange(seq_len)[:, None]
    j = ops.arange(seq_len)[None, :]
    mask = ops.cast(j > i, "float32") * -1e9
    return mask[None, None, :, :]


def whisper_decoder(
    hidden_dim,
    max_target_positions,
    vocab_size,
    decoder_num_layers,
    decoder_attention_heads,
    decoder_ffn_dim,
    activation=_gelu,
    layer_norm_eps=1e-5,
    scale_embedding=False,
    name="decoder",
):
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
    if scale_embedding:
        scale = float(hidden_dim) ** 0.5
        x = layers.Lambda(lambda t, s=scale: t * s, name="decoder_embed_scale")(x)
    x = WhisperLearnedPositionEmbedding(
        max_target_positions=max_target_positions,
        hidden_dim=hidden_dim,
        name="decoder_embed_positions",
    )(x)

    causal_mask = layers.Lambda(
        make_causal_mask_from_ids,
        name="decoder_causal_mask",
        output_shape=lambda s: (1, 1, s[1], s[1]),
    )(decoder_input_ids)

    for i in range(decoder_num_layers):
        x = whisper_decoder_block(
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

    embed_weight = tok_embed.embeddings
    logits = layers.Lambda(
        lambda t, w=embed_weight: ops.matmul(t, ops.transpose(w, (1, 0))),
        name="lm_head",
    )(x)

    return keras.Model(
        inputs={
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        },
        outputs=logits,
        name=name,
    )


@keras.saving.register_keras_serializable(package="kerasformers")
class WhisperModel(FunctionalBaseModel):
    """Whisper encoder-decoder transformer for ASR / translation.

    Wires :func:`whisper_encoder` and :func:`whisper_decoder` into a single
    Functional graph so the full model can be called with one dict:

    >>> out = model({"input_features": mel, "decoder_input_ids": ids})
    >>> out["encoder_hidden_states"]   # (B, T, hidden_dim)
    >>> out["logits"]                  # (B, L, vocab_size)

    This is the teacher-forced training path. For autoregressive
    inference use :class:`WhisperSpeechToText`, which subclasses this
    and adds a ``.generate(audio, processor, ...)`` method.

    Construction:

    >>> WhisperModel.from_weights("whisper_tiny")             # kerasformers release
    >>> WhisperModel.from_weights("hf:openai/whisper-tiny")   # canonical checkpoint
    >>> WhisperModel.from_weights("hf:user/whisper-finetune") # any fine-tune

    .. note::
        Unlike vision models in kerasformers, Whisper has a **fixed input
        shape** dictated by the audio pipeline: log-mel spectrograms
        of ``(num_mel_bins, max_source_positions * 2)`` —
        ``(80, 3000)`` for v1/v2 variants, ``(128, 3000)`` for
        large-v3 / large-v3-turbo. There is no ``input_shape`` kwarg;
        feed :class:`WhisperFeatureExtractor` output directly.

    Args:
        hidden_dim: Hidden / embedding dimension.
        encoder_num_layers: Number of encoder transformer blocks.
        decoder_num_layers: Number of decoder transformer blocks.
        encoder_attention_heads: Encoder self-attn head count.
        decoder_attention_heads: Decoder self-attn / cross-attn head count.
        encoder_ffn_dim: Encoder MLP hidden dim.
        decoder_ffn_dim: Decoder MLP hidden dim.
        num_mel_bins: Mel bin count of the input log-mel spectrogram.
        max_source_positions: Max encoder position. Always ``1500``.
        max_target_positions: Max decoded length. Always ``448``.
        vocab_size: Token vocabulary size.
        activation_function: MLP activation. ``"gelu"`` (exact GELU,
            default, matches OpenAI), ``"gelu_new"`` (tanh-approx),
            ``"relu"``, ``"silu"`` / ``"swish"``.
        layer_norm_eps: Epsilon for every LayerNorm. Defaults to ``1e-5``.
        scale_embedding: Whether to scale the decoder token embedding by
            ``sqrt(hidden_dim)``. ``False`` for canonical OpenAI Whisper.
        name: Model name. Defaults to ``"WhisperModel"``.
    """

    BASE_MODEL_CONFIG = WHISPER_CONFIG
    BASE_WEIGHT_CONFIG = WHISPER_WEIGHTS
    HF_MODEL_TYPE = "whisper"

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
            "num_mel_bins": hf_config.get("num_mel_bins", 80),
            "max_source_positions": hf_config.get("max_source_positions", 1500),
            "max_target_positions": hf_config.get("max_target_positions", 448),
            "vocab_size": hf_config["vocab_size"],
            "activation_function": hf_config.get("activation_function", "gelu"),
            "scale_embedding": hf_config.get("scale_embedding", False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.whisper.convert_whisper_hf_to_keras import (
            transfer_whisper_weights,
        )

        transfer_whisper_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        hidden_dim=384,
        encoder_num_layers=4,
        decoder_num_layers=4,
        encoder_attention_heads=6,
        decoder_attention_heads=6,
        encoder_ffn_dim=1536,
        decoder_ffn_dim=1536,
        num_mel_bins=80,
        max_source_positions=1500,
        max_target_positions=448,
        vocab_size=51865,
        activation_function="gelu",
        layer_norm_eps=1e-5,
        scale_embedding=False,
        name="WhisperModel",
        **kwargs,
    ):
        activation_fn = _resolve_activation(activation_function)

        encoder = whisper_encoder(
            hidden_dim=hidden_dim,
            num_mel_bins=num_mel_bins,
            max_source_positions=max_source_positions,
            encoder_num_layers=encoder_num_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            activation=activation_fn,
            layer_norm_eps=layer_norm_eps,
            name=f"{name}_encoder",
        )
        decoder = whisper_decoder(
            hidden_dim=hidden_dim,
            max_target_positions=max_target_positions,
            vocab_size=vocab_size,
            decoder_num_layers=decoder_num_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            activation=activation_fn,
            layer_norm_eps=layer_norm_eps,
            scale_embedding=scale_embedding,
            name=f"{name}_decoder",
        )

        input_features = layers.Input(shape=(num_mel_bins, None), name="input_features")
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
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.vocab_size = vocab_size
        self.activation_function = activation_function
        self.layer_norm_eps = layer_norm_eps
        self.scale_embedding = scale_embedding

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
                "num_mel_bins": self.num_mel_bins,
                "max_source_positions": self.max_source_positions,
                "max_target_positions": self.max_target_positions,
                "vocab_size": self.vocab_size,
                "activation_function": self.activation_function,
                "layer_norm_eps": self.layer_norm_eps,
                "scale_embedding": self.scale_embedding,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class WhisperSpeechToText(WhisperModel):
    """Whisper speech-to-text model (transcription + translation).

    Composes the same encoder + decoder + tied LM head Functional graph as
    :class:`WhisperModel` (so it loads the same weights and is a drop-in
    replacement for teacher-forced training and forward passes), and adds
    :meth:`generate` — an end-to-end audio → text method that pulls in a
    :class:`~kerasformers.models.whisper.WhisperProcessor` for feature
    extraction, prompt construction, and detokenization.

    This mirrors the reference pattern (``WhisperModel`` is the bare
    encoder/decoder, ``WhisperForConditionalGeneration`` adds the LM head
    + ``.generate()``) and the kerasformers detection-style pattern
    (``DetrModel`` + ``DETRDetect``).

    .. code-block:: python

        model = WhisperSpeechToText.from_weights("whisper_tiny")
        processor = WhisperProcessor.from_weights("whisper_tiny")
        text = model.generate(audio, processor, language="en", task="transcribe")
    """

    def generate(
        self,
        audio,
        processor,
        language: Optional[str] = "en",
        task: str = "transcribe",
        no_timestamps: bool = True,
        max_new_tokens: int = 224,
        sampling_rate: int = 16000,
        return_ids: bool = False,
        suppress_tokens: Optional[List[int]] = None,
        begin_suppress_tokens: Optional[List[int]] = None,
    ) -> Union[List[str], List[List[int]]]:
        inputs = processor(audio=audio, sampling_rate=sampling_rate)
        forced = dict(
            processor.get_decoder_prompt_ids(
                language=language, task=task, no_timestamps=no_timestamps
            )
        )
        decoder_start_token_id = processor.decoder_start_token_id
        eos_token_id = processor.tokenizer.eos_token_id

        suppress_set = set(
            suppress_tokens if suppress_tokens is not None else WHISPER_SUPPRESS_TOKENS
        )
        begin_suppress_set = set(
            begin_suppress_tokens
            if begin_suppress_tokens is not None
            else WHISPER_BEGIN_SUPPRESS_TOKENS
        )

        enc_out = self.encoder(inputs["input_features"])
        batch = enc_out.shape[0]

        suppress_ids = sorted(suppress_set)
        begin_ids = sorted(begin_suppress_set)
        generated = ops.full((batch, 1), decoder_start_token_id, dtype="int32")
        done = ops.zeros((batch,), dtype="bool")
        suppress_bias = None

        for step in range(max_new_tokens):
            cur_pos = generated.shape[1]
            if cur_pos in forced:
                next_ids = ops.full((batch,), forced[cur_pos], dtype="int32")
            else:
                logits = self.decoder(
                    {
                        "decoder_input_ids": generated,
                        "encoder_hidden_states": enc_out,
                    }
                )
                next_logits = logits[:, -1, :]
                vocab = next_logits.shape[-1]
                if suppress_ids:
                    if suppress_bias is None:
                        idx = ops.convert_to_tensor(suppress_ids, dtype="int32")
                        suppress_bias = ops.sum(ops.one_hot(idx, vocab), axis=0) * -1e9
                    next_logits = next_logits + suppress_bias
                if step == 0 and begin_ids:
                    idx = ops.convert_to_tensor(begin_ids, dtype="int32")
                    next_logits = (
                        next_logits + ops.sum(ops.one_hot(idx, vocab), axis=0) * -1e9
                    )
                next_ids = ops.cast(ops.argmax(next_logits, axis=-1), "int32")

            next_ids = ops.cast(ops.where(done, eos_token_id, next_ids), "int32")
            generated = ops.concatenate([generated, next_ids[:, None]], axis=1)
            done = ops.logical_or(done, ops.equal(next_ids, eos_token_id))
            if bool(ops.all(done)):
                break

        ids = [list(row) for row in ops.convert_to_numpy(generated)]
        if return_ids:
            return ids
        return processor.batch_decode(ids, skip_special_tokens=True)


@keras.saving.register_keras_serializable(package="kerasformers")
class WhisperAudioClassify(FunctionalBaseModel):
    """Whisper encoder + linear classifier for audio classification.

    Uses **only the
    Whisper encoder** (no decoder), then a per-frame ``projector`` Dense,
    a mean pool over time, and a final linear classifier producing
    ``num_classes`` logits.

    When ``use_weighted_layer_sum=True``, all encoder hidden states
    (post-embedding through final LayerNorm) are stacked and combined
    by a learnable softmax weighting before the projector — matching
    the SUPERB-style classification head.

    .. code-block:: python

        model = WhisperAudioClassify.from_weights(
            "hf:sanchit-gandhi/whisper-tiny-ft-keyword-spotting"
        )
        processor = WhisperFeatureExtractor()
        mel = processor(audio)
        logits = model(mel)              # (B, num_classes)

    Args:
        hidden_dim: Encoder hidden dimension.
        encoder_num_layers: Number of encoder transformer blocks.
        encoder_attention_heads: Encoder self-attention head count.
        encoder_ffn_dim: Encoder MLP intermediate dim.
        num_mel_bins: Mel bin count of the input log-mel spectrogram.
        max_source_positions: Max encoder position. Always ``1500``.
        num_classes: Number of output classes.
        classifier_proj_size: Projector hidden dim. Defaults to ``256``.
        use_weighted_layer_sum: Combine all encoder hidden states via a
            learnable softmax. Defaults to ``False``.
        activation_function: MLP activation. Defaults to ``"gelu"``.
        layer_norm_eps: Epsilon for every LayerNorm. Defaults to ``1e-5``.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "whisper"

    @classmethod
    def config_from_hf(cls, hf_config):
        from kerasformers.base.base_model import hf_num_classes

        return {
            "hidden_dim": hf_config["d_model"],
            "encoder_num_layers": hf_config["encoder_layers"],
            "encoder_attention_heads": hf_config["encoder_attention_heads"],
            "encoder_ffn_dim": hf_config["encoder_ffn_dim"],
            "num_mel_bins": hf_config.get("num_mel_bins", 80),
            "max_source_positions": hf_config.get("max_source_positions", 1500),
            "num_classes": hf_num_classes(hf_config),
            "classifier_proj_size": hf_config.get("classifier_proj_size", 256),
            "use_weighted_layer_sum": hf_config.get("use_weighted_layer_sum", False),
            "activation_function": hf_config.get("activation_function", "gelu"),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.whisper.convert_whisper_hf_to_keras import (
            transfer_whisper_audio_classify_weights,
        )

        transfer_whisper_audio_classify_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        hidden_dim=384,
        encoder_num_layers=4,
        encoder_attention_heads=6,
        encoder_ffn_dim=1536,
        num_mel_bins=80,
        max_source_positions=1500,
        num_classes=2,
        classifier_proj_size=256,
        use_weighted_layer_sum=False,
        activation_function="gelu",
        layer_norm_eps=1e-5,
        name="WhisperAudioClassify",
        **kwargs,
    ):
        activation_fn = _resolve_activation(activation_function)

        encoder = whisper_encoder(
            hidden_dim=hidden_dim,
            num_mel_bins=num_mel_bins,
            max_source_positions=max_source_positions,
            encoder_num_layers=encoder_num_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            activation=activation_fn,
            layer_norm_eps=layer_norm_eps,
            output_all_hidden_states=use_weighted_layer_sum,
            name=f"{name}_encoder",
        )

        input_features = layers.Input(shape=(num_mel_bins, None), name="input_features")
        encoder_out = encoder(input_features)

        if use_weighted_layer_sum:
            x = WhisperLayerWeights(
                num_layers=encoder_num_layers + 1, name="layer_weights"
            )(encoder_out)
        else:
            x = encoder_out

        x = layers.Dense(classifier_proj_size, name="projector")(x)
        x = ops.mean(x, axis=1)
        logits = layers.Dense(num_classes, name="classifier")(x)

        super().__init__(inputs=input_features, outputs=logits, name=name, **kwargs)

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.num_classes = num_classes
        self.classifier_proj_size = classifier_proj_size
        self.use_weighted_layer_sum = use_weighted_layer_sum
        self.activation_function = activation_function
        self.layer_norm_eps = layer_norm_eps

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "encoder_num_layers": self.encoder_num_layers,
                "encoder_attention_heads": self.encoder_attention_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "num_mel_bins": self.num_mel_bins,
                "max_source_positions": self.max_source_positions,
                "num_classes": self.num_classes,
                "classifier_proj_size": self.classifier_proj_size,
                "use_weighted_layer_sum": self.use_weighted_layer_sum,
                "activation_function": self.activation_function,
                "layer_norm_eps": self.layer_norm_eps,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
