import keras
from keras import layers, ops

from kerasformers.base import (
    BaseSeq2SeqGeneration,
    CheckpointSource,
    SubclassedBaseModel,
)

from .t5_config import T5Config
from .t5_layers import (
    MASK_NEG,
    T5DecoderBlock,
    T5EncoderBlock,
    T5LayerNorm,
    compute_relative_bias,
)

# All T5 classes (backbone + generative head + encoder + task heads) share the variant's
# weights repo, whose kf_config.json declares the canonical T5Model. The single hosted
# checkpoint is the full encoder-decoder; each class loads its own subset.
T5_HUB_SIBLINGS = frozenset(
    {
        "T5Model",
        "T5ConditionalGenerate",
        "T5EncoderModel",
        "T5SequenceClassify",
        "T5TokenClassify",
        "T5QnA",
    }
)

_ARCH_FIELDS = (
    "vocab_size",
    "embed_dim",
    "key_value_dim",
    "mlp_dim",
    "num_layers",
    "num_decoder_layers",
    "num_heads",
    "relative_attention_num_buckets",
    "relative_attention_max_distance",
    "hidden_act",
    "layer_norm_eps",
    "dropout",
    "tie_word_embeddings",
    "pad_token_id",
    "eos_token_id",
    "decoder_start_token_id",
)


def t5_config_from_hf(hf_config):
    return {
        "vocab_size": hf_config["vocab_size"],
        "embed_dim": hf_config["d_model"],
        "key_value_dim": hf_config["d_kv"],
        "mlp_dim": hf_config["d_ff"],
        "num_layers": hf_config["num_layers"],
        "num_decoder_layers": hf_config.get("num_decoder_layers")
        or hf_config["num_layers"],
        "num_heads": hf_config["num_heads"],
        "relative_attention_num_buckets": hf_config.get(
            "relative_attention_num_buckets", 32
        ),
        "relative_attention_max_distance": hf_config.get(
            "relative_attention_max_distance", 128
        ),
        "hidden_act": hf_config.get("dense_act_fn", "relu"),
        "layer_norm_eps": hf_config.get("layer_norm_epsilon", 1e-6),
        "dropout": hf_config.get("dropout_rate", 0.1),
        "tie_word_embeddings": hf_config.get("tie_word_embeddings", True),
        "pad_token_id": hf_config.get("pad_token_id", 0),
        "eos_token_id": hf_config.get("eos_token_id", 1),
        "decoder_start_token_id": hf_config.get("decoder_start_token_id", 0),
    }


def padding_bias(attention_mask, batch, seq):
    """Additive (batch, 1, 1, seq) mask: 0 to keep, MASK_NEG to drop."""
    if attention_mask is None:
        return ops.zeros((batch, 1, 1, seq), dtype="float32")
    am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
    return (1.0 - am)[:, None, None, :] * MASK_NEG


def causal_bias(seq):
    qi = ops.arange(seq)[:, None]
    ki = ops.arange(seq)[None, :]
    return ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]


class _T5Encoder:
    """Shared encoder forward (mixed into T5Model / T5EncoderModel)."""

    def build_encoder(self):
        self.encoder_rel_bias = layers.Embedding(
            self.relative_attention_num_buckets,
            self.num_heads,
            name="encoder_rel_bias",
        )
        self.encoder_blocks = [
            T5EncoderBlock(
                self.embed_dim,
                self.key_value_dim,
                self.num_heads,
                self.mlp_dim,
                self.hidden_act,
                self.layer_norm_eps,
                prefix=f"enc_{i}",
                name=f"encoder_block_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.encoder_final_layer_norm = T5LayerNorm(
            self.layer_norm_eps, name="encoder_final_layer_norm"
        )

    def encode(self, input_ids, attention_mask=None):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        hidden = self.shared(input_ids)
        position_bias = compute_relative_bias(
            self.encoder_rel_bias,
            seq,
            seq,
            True,
            self.relative_attention_num_buckets,
            self.relative_attention_max_distance,
        ) + padding_bias(attention_mask, batch, seq)
        for block in self.encoder_blocks:
            hidden = block(hidden, position_bias)
        return self.encoder_final_layer_norm(hidden)


@keras.saving.register_keras_serializable(package="kerasformers")
class T5Model(SubclassedBaseModel, _T5Encoder):
    """Original T5 encoder-decoder backbone (no LM head).

    A shared token embedding feeds a bidirectional encoder and a causal decoder that
    cross-attends to the encoder output. Attention uses learned relative position bias
    (shared within each stack), T5-style RMSNorm (``T5LayerNorm``), pre-LayerNorm
    residuals, and bias-free projections. Subclassed (imperative) model run with
    ``keras.ops``. Returns the decoder ``last_hidden_state`` (and the encoder output);
    use :class:`T5ConditionalGenerate` for logits / text.

    References:
    - [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model dimension (``d_model``).
        key_value_dim: Per-head q/k/v dimension (``d_kv``).
        mlp_dim: Feed-forward intermediate size (``d_ff``).
        num_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        relative_attention_num_buckets: Relative-position-bias bucket count.
        relative_attention_max_distance: Maximum distance for the relative bias.
        hidden_act: Feed-forward activation.
        layer_norm_eps: RMSNorm epsilon.
        dropout: Dropout rate (inference: unused).
        tie_word_embeddings: Whether :class:`T5ConditionalGenerate` ties + scales the LM head.
        pad_token_id: Padding token id (also the decoder start token).
        eos_token_id: End-of-sequence token id.
        decoder_start_token_id: First decoder input token.
    """

    HF_MODEL_TYPE = "t5"
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = T5Config
    HUB_REPO_SIBLINGS = T5_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("T5Model")

    def __init__(
        self,
        vocab_size=32128,
        embed_dim=768,
        key_value_dim=64,
        mlp_dim=3072,
        num_layers=12,
        num_decoder_layers=12,
        num_heads=12,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        hidden_act="relu",
        layer_norm_eps=1e-6,
        dropout=0.1,
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.key_value_dim = key_value_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id

        self.shared = layers.Embedding(vocab_size, embed_dim, name="shared")
        self.build_encoder()
        self.decoder_rel_bias = layers.Embedding(
            relative_attention_num_buckets, num_heads, name="decoder_rel_bias"
        )
        self.decoder_blocks = [
            T5DecoderBlock(
                embed_dim,
                key_value_dim,
                num_heads,
                mlp_dim,
                hidden_act,
                layer_norm_eps,
                prefix=f"dec_{i}",
                name=f"decoder_block_{i}",
            )
            for i in range(num_decoder_layers)
        ]
        self.decoder_final_layer_norm = T5LayerNorm(
            layer_norm_eps, name="decoder_final_layer_norm"
        )

    def shift_right(self, input_ids):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch = int(input_ids.shape[0])
        start = ops.full((batch, 1), self.decoder_start_token_id, dtype="int32")
        return ops.concatenate([start, input_ids[:, :-1]], axis=1)

    def decode(
        self,
        decoder_input_ids,
        encoder_hidden_states,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
    ):
        decoder_input_ids = ops.cast(ops.convert_to_tensor(decoder_input_ids), "int32")
        batch, dseq = int(decoder_input_ids.shape[0]), int(decoder_input_ids.shape[1])
        eseq = int(encoder_hidden_states.shape[1])
        hidden = self.shared(decoder_input_ids)

        self_bias = (
            compute_relative_bias(
                self.decoder_rel_bias,
                dseq,
                dseq,
                False,
                self.relative_attention_num_buckets,
                self.relative_attention_max_distance,
            )
            + causal_bias(dseq)
            + padding_bias(decoder_attention_mask, batch, dseq)
        )
        cross_bias = padding_bias(encoder_attention_mask, batch, eseq)
        for block in self.decoder_blocks:
            hidden = block(hidden, self_bias, encoder_hidden_states, cross_bias)
        return self.decoder_final_layer_norm(hidden)

    def call(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        decoder_input_ids = inputs.get("decoder_input_ids")
        if decoder_input_ids is None:
            decoder_input_ids = self.shift_right(input_ids)
        decoder_attention_mask = inputs.get("decoder_attention_mask")
        encoder_hidden_states = self.encode(input_ids, attention_mask)
        decoder_hidden = self.decode(
            decoder_input_ids,
            encoder_hidden_states,
            decoder_attention_mask,
            attention_mask,
        )
        return {
            "last_hidden_state": decoder_hidden,
            "encoder_last_hidden_state": encoder_hidden_states,
        }

    @classmethod
    def config_from_hf(cls, hf_config):
        return t5_config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_t5_hf_to_keras import transfer_t5_weights

        transfer_t5_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update({k: getattr(self, k) for k in _ARCH_FIELDS})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class T5ConditionalGenerate(T5Model, BaseSeq2SeqGeneration):
    """T5 backbone + a (tied, scaled) language-model head and text-to-text generation.

    ``call`` returns ``logits`` ``(batch, target_seq, vocab_size)`` plus the decoder and
    encoder hidden states. For original T5 the LM head is the transposed shared embedding
    and the decoder output is scaled by ``embed_dim ** -0.5`` first (``tie_word_embeddings``).
    ``generate`` runs the encoder once and greedily decodes with the shared embedding as
    the head. Constructor ``Args`` are inherited from :class:`T5Model`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.tie_word_embeddings:
            self.lm_head = layers.Dense(self.vocab_size, use_bias=False, name="lm_head")

    def project(self, hidden):
        if self.tie_word_embeddings:
            hidden = hidden * (self.embed_dim**-0.5)
            return ops.matmul(hidden, ops.transpose(self.shared.embeddings))
        return self.lm_head(hidden)

    def call(self, inputs):
        outputs = super().call(inputs)
        outputs["logits"] = self.project(outputs["last_hidden_state"])
        return outputs

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=None,
        eos_token_id=None,
        **kwargs,
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        if attention_mask is not None:
            attention_mask = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
        max_new_tokens = max_new_tokens or 64
        eos = eos_token_id if eos_token_id is not None else self.eos_token_id
        batch = int(input_ids.shape[0])

        encoder_hidden_states = self.encode(input_ids, attention_mask)
        generated = ops.full((batch, 1), self.decoder_start_token_id, dtype="int32")
        done = ops.zeros((batch,), dtype="bool")
        for _ in range(max_new_tokens):
            decoder_hidden = self.decode(
                generated, encoder_hidden_states, None, attention_mask
            )
            next_logits = self.project(decoder_hidden)[:, -1, :]
            next_ids = ops.cast(ops.argmax(next_logits, axis=-1), "int32")
            next_ids = ops.cast(ops.where(done, eos, next_ids), "int32")
            generated = ops.concatenate([generated, next_ids[:, None]], axis=1)
            done = ops.logical_or(done, ops.equal(next_ids, eos))
            if bool(ops.all(done)):
                break
        return generated


@keras.saving.register_keras_serializable(package="kerasformers")
class T5EncoderModel(SubclassedBaseModel, _T5Encoder):
    """T5 encoder stack only (no decoder, no LM head).

    A shared token embedding feeds the bidirectional encoder; returns the encoder
    ``last_hidden_state`` ``(batch, seq, embed_dim)`` for embedding / feature use.
    Constructor ``Args`` mirror :class:`T5Model` (decoder fields are accepted and ignored).
    """

    HF_MODEL_TYPE = "t5"
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = T5Config
    HUB_REPO_SIBLINGS = T5_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("T5Model")

    def __init__(
        self,
        vocab_size=32128,
        embed_dim=768,
        key_value_dim=64,
        mlp_dim=3072,
        num_layers=12,
        num_decoder_layers=12,
        num_heads=12,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        hidden_act="relu",
        layer_norm_eps=1e-6,
        dropout=0.1,
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.key_value_dim = key_value_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id

        self.shared = layers.Embedding(vocab_size, embed_dim, name="shared")
        self.build_encoder()

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        hidden = self.encode(inputs["input_ids"], inputs.get("attention_mask"))
        return {"last_hidden_state": hidden}

    @classmethod
    def config_from_hf(cls, hf_config):
        return t5_config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_t5_hf_to_keras import transfer_t5_weights

        transfer_t5_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update({k: getattr(self, k) for k in _ARCH_FIELDS})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class T5SequenceClassify(T5Model):
    """T5 sequence classifier (HF ``T5ForSequenceClassification``).

    The full encoder-decoder backbone (decoder input = right-shifted ``input_ids``), then
    the decoder hidden state at the last EOS position is passed to a tanh classification
    head. Returns logits ``(batch, num_classes)``. Constructor ``Args`` extend
    :class:`T5Model` with ``num_classes`` and ``classifier_dropout``.
    """

    def __init__(self, num_classes=2, classifier_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.classifier_dense = layers.Dense(self.embed_dim, name="classifier_dense")
        self.classifier_out_proj = layers.Dense(num_classes, name="classifier_out_proj")

    def call(self, inputs):
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        attention_mask = inputs.get("attention_mask")
        sequence_output = self.decode(
            self.shift_right(input_ids),
            self.encode(input_ids, attention_mask),
            None,
            attention_mask,
        )
        seq = int(input_ids.shape[1])
        eos_positions = ops.where(
            ops.equal(input_ids, self.eos_token_id), ops.arange(seq)[None, :], -1
        )
        last_eos = ops.max(eos_positions, axis=1)
        selected = ops.take_along_axis(
            sequence_output, last_eos[:, None, None], axis=1
        )[:, 0, :]
        return self.classifier_out_proj(ops.tanh(self.classifier_dense(selected)))

    @classmethod
    def config_from_hf(cls, hf_config):
        config = t5_config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        return config

    def get_config(self):
        config = super().get_config()
        config["num_classes"] = self.num_classes
        config["classifier_dropout"] = self.classifier_dropout
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class T5TokenClassify(T5EncoderModel):
    """T5 token classifier (HF ``T5ForTokenClassification``).

    The ENCODER only, then a per-token linear classifier. Returns logits
    ``(batch, seq, num_classes)``. Constructor ``Args`` extend :class:`T5EncoderModel`
    with ``num_classes`` and ``classifier_dropout``.
    """

    def __init__(self, num_classes=2, classifier_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.classifier = layers.Dense(num_classes, name="classifier")

    def call(self, inputs):
        hidden = self.encode(inputs["input_ids"], inputs.get("attention_mask"))
        return self.classifier(hidden)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = t5_config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        return config

    def get_config(self):
        config = super().get_config()
        config["num_classes"] = self.num_classes
        config["classifier_dropout"] = self.classifier_dropout
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class T5QnA(T5Model):
    """T5 extractive question answering (HF ``T5ForQuestionAnswering``).

    The full encoder-decoder backbone (decoder input = right-shifted ``input_ids``) and a
    linear ``qa_outputs`` head over the decoder output, returning ``start_logits`` and
    ``end_logits`` ``(batch, seq)`` each. Constructor ``Args`` are inherited from
    :class:`T5Model`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.qa_outputs = layers.Dense(2, name="qa_outputs")

    def call(self, inputs):
        input_ids = ops.cast(ops.convert_to_tensor(inputs["input_ids"]), "int32")
        attention_mask = inputs.get("attention_mask")
        sequence_output = self.decode(
            self.shift_right(input_ids),
            self.encode(input_ids, attention_mask),
            None,
            attention_mask,
        )
        start_logits, end_logits = ops.split(
            self.qa_outputs(sequence_output), 2, axis=-1
        )
        return {
            "start_logits": ops.squeeze(start_logits, -1),
            "end_logits": ops.squeeze(end_logits, -1),
        }
