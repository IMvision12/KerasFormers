import keras
from keras import layers, ops

from kerasformers.base import BaseModel
from kerasformers.weight_utils import copy_weights_by_path_suffix

from .config import DISTILBERT_MODEL_CONFIG, DISTILBERT_WEIGHT_CONFIG
from .distilbert_layers import (
    DistilBertEmbeddings,
    DistilBertFlattenChoices,
    DistilBertSelfAttention,
    DistilBertUnflattenChoices,
)

BASE_MODEL_CONFIG = {
    v: DISTILBERT_MODEL_CONFIG[m["model"]] for v, m in DISTILBERT_WEIGHT_CONFIG.items()
}
MLM_WEIGHT_CONFIG = {
    v: {**m, "url": m["mlm_url"]} for v, m in DISTILBERT_WEIGHT_CONFIG.items()
}


def distilbert_encoder_layer(
    x,
    attention_mask,
    *,
    embed_dim,
    num_heads,
    mlp_dim,
    hidden_act,
    layer_norm_eps,
    dropout,
    attention_dropout,
    layer_idx,
):
    """One DistilBERT block: self-attention + feed-forward, both post-LayerNorm.

    Matches Hugging Face: ``sa_layer_norm(attn(x) + x)`` then
    ``output_layer_norm(ffn(.) + .)``. The attention output projection lives
    inside :class:`DistilBertSelfAttention` (``out_lin``).
    """
    prefix = f"blocks_{layer_idx}"

    attn = DistilBertSelfAttention(
        embed_dim,
        num_heads,
        attention_dropout=attention_dropout,
        block_prefix=prefix,
        name=f"{prefix}_attention",
    )(x, attention_mask=attention_mask)
    attn = layers.Add(name=f"{prefix}_sa_add")([attn, x])
    attn = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_sa_layer_norm"
    )(attn)

    ffn = layers.Dense(mlp_dim, name=f"{prefix}_ffn_lin1")(attn)
    ffn = layers.Activation(hidden_act, name=f"{prefix}_ffn_act")(ffn)
    ffn = layers.Dense(embed_dim, name=f"{prefix}_ffn_lin2")(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Add(name=f"{prefix}_output_add")([ffn, attn])
    ffn = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_output_layer_norm"
    )(ffn)
    return ffn


def distilbert_backbone(
    input_ids,
    attention_mask,
    *,
    vocab_size,
    embed_dim,
    num_layers,
    num_heads,
    mlp_dim,
    max_position_embeddings,
    hidden_act,
    layer_norm_eps,
    dropout,
    attention_dropout,
):
    """DistilBERT embeddings + transformer encoder.

    Returns the last block's hidden states ``(B, seq, embed_dim)`` — DistilBERT
    has no pooler and no final norm.
    """
    x = DistilBertEmbeddings(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_position_embeddings=max_position_embeddings,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        name="embeddings",
    )(input_ids)

    mask = ops.cast(attention_mask, "float32")
    mask = ops.expand_dims(ops.expand_dims(mask, 1), 1)
    mask = (1.0 - mask) * -1e9

    for i in range(num_layers):
        x = distilbert_encoder_layer(
            x,
            mask,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            attention_dropout=attention_dropout,
            layer_idx=i,
        )
    return x


@keras.saving.register_keras_serializable(package="kerasformers")
class DistilBertModel(BaseModel):
    """Instantiates the DistilBERT encoder backbone.

    DistilBERT embeds tokens with summed word + absolute-position embeddings (no
    token-type / segment embeddings), then applies a stack of bidirectional
    transformer encoder layers (multi-head self-attention + feed-forward, each
    with a post-LayerNorm residual). There is no pooler.

    The model takes a dict of ``input_ids`` and ``attention_mask`` (both
    ``(B, seq)`` int tensors, as produced by :class:`DistilBertTokenizer`) and
    returns a dict with ``last_hidden_state`` ``(B, seq, embed_dim)``.

    References:
    - [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)

    Args:
        vocab_size: Integer, token vocabulary size. Defaults to `30522`.
        embed_dim: Integer, model / embedding dimension. Defaults to `768`.
        num_layers: Integer, number of transformer encoder layers. Defaults to `6`.
        num_heads: Integer, number of attention heads. Defaults to `12`.
        mlp_dim: Integer, feed-forward hidden dimension. Defaults to `3072`.
        max_position_embeddings: Integer, size of the position-embedding table.
            Defaults to `512`.
        hidden_act: String, feed-forward activation. Defaults to `"gelu"`.
        layer_norm_eps: Float, LayerNorm epsilon. Defaults to `1e-12`.
        pad_token_id: Integer, padding token id. Defaults to `0`.
        dropout: Float, hidden dropout rate. Defaults to `0.0`.
        attention_dropout: Float, attention-weight dropout rate. Defaults to `0.0`.
        name: String, model name. Defaults to `"DistilBertModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = DISTILBERT_WEIGHT_CONFIG
    HF_MODEL_TYPE = "distilbert"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_distilbert_hf_to_keras import transfer_distilbert_weights

        transfer_distilbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["dim"],
            "num_layers": hf_config["n_layers"],
            "num_heads": hf_config["n_heads"],
            "mlp_dim": hf_config["hidden_dim"],
            "max_position_embeddings": hf_config["max_position_embeddings"],
            "hidden_act": hf_config.get("activation", "gelu"),
            "pad_token_id": hf_config.get("pad_token_id", 0),
        }

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        name="DistilBertModel",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        sequence_output = distilbert_backbone(
            inputs["input_ids"],
            inputs["attention_mask"],
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        super().__init__(
            inputs=inputs,
            outputs={"last_hidden_state": sequence_output},
            name=name,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class DistilBertMaskedLM(BaseModel):
    """DistilBERT with the masked-language-modeling head.

    Wraps a :class:`DistilBertModel` backbone and attaches DistilBERT's MLM head
    — ``vocab_transform`` (dense) + ``hidden_act`` + ``vocab_layer_norm``, then
    the ``vocab_projector`` decoder — producing token logits
    ``(B, seq, vocab_size)``. The head weights are part of the pretrained
    checkpoint, so ``from_weights`` restores a ready-to-use fill-mask model.

    References:
    - [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)

    Args:
        See :class:`DistilBertModel` for the backbone arguments.
        name: String, model name. Defaults to `"DistilBertMaskedLM"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = MLM_WEIGHT_CONFIG
    HF_MODEL_TYPE = "distilbert"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_distilbert_hf_to_keras import transfer_distilbert_weights

        transfer_distilbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DistilBertModel.config_from_hf(hf_config)

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        name="DistilBertMaskedLM",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)

        backbone = DistilBertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        x = layers.Dense(embed_dim, name="vocab_transform")(x)
        x = layers.Activation(hidden_act, name="vocab_transform_act")(x)
        x = layers.LayerNormalization(epsilon=layer_norm_eps, name="vocab_layer_norm")(
            x
        )
        logits = layers.Dense(vocab_size, name="vocab_projector")(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class DistilBertSequenceClassify(BaseModel):
    """DistilBERT sentence/sequence classifier.

    Wraps a :class:`DistilBertModel` backbone and attaches DistilBERT's
    classification head on the first ([CLS]) token: ``pre_classifier`` dense +
    ReLU + dropout + a dense classifier, producing ``num_classes`` logits
    ``(B, num_classes)``. The head is randomly initialized from the pretrained
    checkpoint and meant for fine-tuning.

    References:
    - [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)

    Args:
        See :class:`DistilBertModel` for the backbone arguments.
        num_classes: Integer, number of output classes. Defaults to `2`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        name: String, model name. Defaults to `"DistilBertSequenceClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = DISTILBERT_WEIGHT_CONFIG
    HF_MODEL_TYPE = "distilbert"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_distilbert_hf_to_keras import transfer_distilbert_weights

        transfer_distilbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = DistilBertModel.config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        return config

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = DistilBertModel.from_weights(variant, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=2,
        classifier_dropout=0.0,
        name="DistilBertSequenceClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url"):
            kwargs.pop(k, None)

        backbone = DistilBertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"][:, 0]
        x = layers.Dense(embed_dim, name="pre_classifier")(x)
        x = layers.Activation("relu", name="pre_classifier_act")(x)
        x = layers.Dropout(classifier_dropout)(x)
        logits = layers.Dense(num_classes, name="classifier")(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "num_classes": self.num_classes,
                "classifier_dropout": self.classifier_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class DistilBertTokenClassify(BaseModel):
    """DistilBERT token classifier (e.g. NER / POS tagging).

    Wraps a :class:`DistilBertModel` backbone and attaches dropout plus a dense
    head applied per token, producing ``num_classes`` logits
    ``(B, seq, num_classes)``. The head is randomly initialized and meant for
    fine-tuning.

    References:
    - [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)

    Args:
        See :class:`DistilBertModel` for the backbone arguments.
        num_classes: Integer, number of token classes. Defaults to `2`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        name: String, model name. Defaults to `"DistilBertTokenClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = DISTILBERT_WEIGHT_CONFIG
    HF_MODEL_TYPE = "distilbert"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_distilbert_hf_to_keras import transfer_distilbert_weights

        transfer_distilbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = DistilBertModel.config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        return config

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = DistilBertModel.from_weights(variant, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=2,
        classifier_dropout=0.0,
        name="DistilBertTokenClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url"):
            kwargs.pop(k, None)

        backbone = DistilBertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        x = layers.Dropout(classifier_dropout)(x)
        logits = layers.Dense(num_classes, name="classifier")(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "num_classes": self.num_classes,
                "classifier_dropout": self.classifier_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class DistilBertQnA(BaseModel):
    """DistilBERT extractive question-answering head.

    Wraps a :class:`DistilBertModel` backbone and attaches dropout plus a dense
    span head that maps each token to two logits, split into ``start_logits`` and
    ``end_logits`` (each ``(B, seq)``). The head is randomly initialized and meant
    for fine-tuning (or loaded from a fine-tuned ``hf:`` repo such as a SQuAD
    model).

    References:
    - [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)

    Args:
        See :class:`DistilBertModel` for the backbone arguments.
        classifier_dropout: Float, dropout before the span head. Defaults to `0.0`.
        name: String, model name. Defaults to `"DistilBertQnA"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = DISTILBERT_WEIGHT_CONFIG
    HF_MODEL_TYPE = "distilbert"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_distilbert_hf_to_keras import transfer_distilbert_weights

        transfer_distilbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DistilBertModel.config_from_hf(hf_config)

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = DistilBertModel.from_weights(variant, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        name="DistilBertQnA",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)

        backbone = DistilBertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        x = layers.Dropout(classifier_dropout)(x)
        span = layers.Dense(2, name="qa_outputs")(x)
        outputs = {"start_logits": span[:, :, 0], "end_logits": span[:, :, 1]}

        super().__init__(inputs=backbone.input, outputs=outputs, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.classifier_dropout = classifier_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "classifier_dropout": self.classifier_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class DistilBertMultipleChoice(BaseModel):
    """DistilBERT multiple-choice head (e.g. SWAG).

    Takes a dict of ``(B, num_choices, seq)`` int tensors, flattens the choices
    into the batch, runs the :class:`DistilBertModel` backbone, scores each choice
    from its first ([CLS]) token via ``pre_classifier`` + ReLU + dropout + a
    shared dense layer, and reshapes back to per-example ``(B, num_choices)``
    logits. The head is randomly initialized and meant for fine-tuning.

    References:
    - [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)

    Args:
        See :class:`DistilBertModel` for the backbone arguments.
        classifier_dropout: Float, dropout before the choice scorer. Defaults to `0.0`.
        name: String, model name. Defaults to `"DistilBertMultipleChoice"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = DISTILBERT_WEIGHT_CONFIG
    HF_MODEL_TYPE = "distilbert"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_distilbert_hf_to_keras import transfer_distilbert_weights

        transfer_distilbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DistilBertModel.config_from_hf(hf_config)

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = DistilBertModel.from_weights(variant, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        num_choices=4,
        classifier_dropout=0.0,
        name="DistilBertMultipleChoice",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)

        input_ids = layers.Input(
            shape=(num_choices, None), dtype="int32", name="input_ids"
        )
        attention_mask = layers.Input(
            shape=(num_choices, None), dtype="int32", name="attention_mask"
        )

        backbone = DistilBertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        flatten = DistilBertFlattenChoices(name="flatten_choices")
        sequence_output = backbone(
            {
                "input_ids": flatten(input_ids),
                "attention_mask": flatten(attention_mask),
            }
        )["last_hidden_state"]
        x = sequence_output[:, 0]
        x = layers.Dense(embed_dim, name="pre_classifier")(x)
        x = layers.Activation("relu", name="pre_classifier_act")(x)
        x = layers.Dropout(classifier_dropout)(x)
        x = layers.Dense(1, name="classifier")(x)
        logits = DistilBertUnflattenChoices(num_choices, name="unflatten_choices")(x)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        super().__init__(inputs=inputs, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_choices = num_choices
        self.classifier_dropout = classifier_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "num_choices": self.num_choices,
                "classifier_dropout": self.classifier_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
