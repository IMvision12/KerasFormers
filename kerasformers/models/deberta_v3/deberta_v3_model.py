import keras
from keras import layers

from kerasformers.base import BaseModel
from kerasformers.models.deberta_v2.convert_deberta_v2_hf_to_keras import (
    transfer_deberta_v2_weights,
)
from kerasformers.models.deberta_v2.deberta_v2_layers import (
    DebertaV2FlattenChoices,
    DebertaV2UnflattenChoices,
)
from kerasformers.models.deberta_v2.deberta_v2_model import (
    DebertaV2Model,
    deberta_v2_backbone,
)
from kerasformers.weight_utils import copy_weights_by_path_suffix

from .config import DEBERTA_V3_MODEL_CONFIG, DEBERTA_V3_WEIGHT_CONFIG

BASE_MODEL_CONFIG = {
    v: DEBERTA_V3_MODEL_CONFIG[m["model"]] for v, m in DEBERTA_V3_WEIGHT_CONFIG.items()
}
MLM_WEIGHT_CONFIG = {
    v: {**m, "url": m["mlm_url"]} for v, m in DEBERTA_V3_WEIGHT_CONFIG.items()
}


@keras.saving.register_keras_serializable(package="kerasformers")
class DebertaV3Model(BaseModel):
    """Instantiates the DeBERTa-v3 encoder backbone.

    DeBERTa-v3 keeps DeBERTa-v2's architecture (log-bucketed disentangled
    attention, shared key/query relative projections, LayerNorm-ed relative
    embeddings) and differs only in pretraining (ELECTRA-style replaced-token
    detection) and tokenizer vocabulary. It reuses the v2 backbone with no
    convolution layer.

    Takes ``input_ids`` / ``attention_mask`` / ``token_type_ids`` and returns
    ``{"last_hidden_state": (B, seq, embed_dim)}``.

    References:
    - [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training](https://arxiv.org/abs/2111.09543)

    Args:
        See :class:`~kerasformers.models.deberta_v2.DebertaV2Model` for the
        backbone arguments. Defaults match ``deberta_v3_base`` (768 / 12 / 12,
        no conv).
        name: String, model name. Defaults to `"DebertaV3Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = DEBERTA_V3_WEIGHT_CONFIG
    HF_MODEL_TYPE = "deberta-v2"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DebertaV2Model.config_from_hf(hf_config)

    def __init__(
        self,
        vocab_size=128100,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        max_relative_positions=512,
        position_buckets=256,
        pos_att_type=("p2c", "c2p"),
        norm_rel_ebd=True,
        conv_kernel_size=0,
        conv_act="gelu",
        hidden_act="gelu",
        layer_norm_eps=1e-7,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        name="DebertaV3Model",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        pos_att_type = list(pos_att_type)

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
            "token_type_ids": layers.Input(
                shape=(None,), dtype="int32", name="token_type_ids"
            ),
        }
        # deberta_v2_backbone() takes every backbone kwarg except the model-level
        # max_position_embeddings / pad_token_id.
        sequence_output = deberta_v2_backbone(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_relative_positions=max_relative_positions,
            position_buckets=position_buckets,
            pos_att_type=pos_att_type,
            norm_rel_ebd=norm_rel_ebd,
            conv_kernel_size=conv_kernel_size,
            conv_act=conv_act,
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
        self.max_relative_positions = max_relative_positions
        self.position_buckets = position_buckets
        self.pos_att_type = pos_att_type
        self.norm_rel_ebd = norm_rel_ebd
        self.conv_kernel_size = conv_kernel_size
        self.conv_act = conv_act
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
                "max_relative_positions": self.max_relative_positions,
                "position_buckets": self.position_buckets,
                "pos_att_type": self.pos_att_type,
                "norm_rel_ebd": self.norm_rel_ebd,
                "conv_kernel_size": self.conv_kernel_size,
                "conv_act": self.conv_act,
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
class DebertaV3MaskedLM(BaseModel):
    """DeBERTa-v3 masked-LM head (dense + gelu + LayerNorm + tied decoder)."""

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = MLM_WEIGHT_CONFIG
    HF_MODEL_TYPE = "deberta-v2"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DebertaV2Model.config_from_hf(hf_config)

    def __init__(self, name="DebertaV3MaskedLM", **kwargs):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "norm_rel_ebd": True,
            "conv_kernel_size": 0,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV3Model(**cfg, name=f"{name}_backbone")
        x = backbone.output["last_hidden_state"]
        x = layers.Dense(cfg["embed_dim"], name="lm_head_dense")(x)
        x = layers.Activation("gelu", name="lm_head_act")(x)
        x = layers.LayerNormalization(
            epsilon=cfg["layer_norm_eps"], name="lm_head_layernorm"
        )(x)
        logits = layers.Dense(cfg["vocab_size"], name="lm_head_decoder")(x)
        super().__init__(inputs=backbone.input, outputs=logits, name=name)
        self._cfg = cfg

    def get_config(self):
        config = super().get_config()
        config.update({**self._cfg, "name": self.name})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _classify_head(
    backbone,
    embed_dim,
    num_classes,
    pooler_dropout,
    classifier_dropout,
    classifier_activation,
):
    x = backbone.output["last_hidden_state"][:, 0]
    x = layers.Dropout(pooler_dropout)(x)
    x = layers.Dense(embed_dim, name="pooler_dense")(x)
    x = layers.Activation("gelu", name="pooler_act")(x)
    x = layers.Dropout(classifier_dropout)(x)
    return layers.Dense(
        num_classes, activation=classifier_activation, name="classifier"
    )(x)


@keras.saving.register_keras_serializable(package="kerasformers")
class DebertaV3SequenceClassify(BaseModel):
    """DeBERTa-v3 sequence classifier (context pooler + dense classifier)."""

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = DEBERTA_V3_WEIGHT_CONFIG
    HF_MODEL_TYPE = "deberta-v2"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = DebertaV2Model.config_from_hf(hf_config)
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
            src = DebertaV3Model.from_weights(variant, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    def __init__(
        self,
        num_classes=2,
        pooler_dropout=0.0,
        classifier_dropout=0.0,
        classifier_activation="linear",
        name="DebertaV3SequenceClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "norm_rel_ebd": True,
            "conv_kernel_size": 0,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV3Model(**cfg, name=f"{name}_backbone")
        logits = _classify_head(
            backbone,
            cfg["embed_dim"],
            num_classes,
            pooler_dropout,
            classifier_dropout,
            classifier_activation,
        )
        super().__init__(inputs=backbone.input, outputs=logits, name=name)
        self._cfg = cfg
        self.num_classes = num_classes
        self.pooler_dropout = pooler_dropout
        self.classifier_dropout = classifier_dropout
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                **self._cfg,
                "num_classes": self.num_classes,
                "pooler_dropout": self.pooler_dropout,
                "classifier_dropout": self.classifier_dropout,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class DebertaV3TokenClassify(BaseModel):
    """DeBERTa-v3 token classifier (dropout + per-token dense head)."""

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = DEBERTA_V3_WEIGHT_CONFIG
    HF_MODEL_TYPE = "deberta-v2"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = DebertaV2Model.config_from_hf(hf_config)
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
            src = DebertaV3Model.from_weights(variant, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    def __init__(
        self,
        num_classes=2,
        classifier_dropout=0.0,
        classifier_activation="linear",
        name="DebertaV3TokenClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "norm_rel_ebd": True,
            "conv_kernel_size": 0,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV3Model(**cfg, name=f"{name}_backbone")
        x = backbone.output["last_hidden_state"]
        x = layers.Dropout(classifier_dropout)(x)
        logits = layers.Dense(
            num_classes, activation=classifier_activation, name="classifier"
        )(x)
        super().__init__(inputs=backbone.input, outputs=logits, name=name)
        self._cfg = cfg
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                **self._cfg,
                "num_classes": self.num_classes,
                "classifier_dropout": self.classifier_dropout,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class DebertaV3QnA(BaseModel):
    """DeBERTa-v3 extractive QA head (dense span -> start/end logits)."""

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = DEBERTA_V3_WEIGHT_CONFIG
    HF_MODEL_TYPE = "deberta-v2"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DebertaV2Model.config_from_hf(hf_config)

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = DebertaV3Model.from_weights(variant, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    def __init__(self, name="DebertaV3QnA", **kwargs):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "norm_rel_ebd": True,
            "conv_kernel_size": 0,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV3Model(**cfg, name=f"{name}_backbone")
        span = layers.Dense(2, name="qa_outputs")(backbone.output["last_hidden_state"])
        outputs = {"start_logits": span[:, :, 0], "end_logits": span[:, :, 1]}
        super().__init__(inputs=backbone.input, outputs=outputs, name=name)
        self._cfg = cfg

    def get_config(self):
        config = super().get_config()
        config.update({**self._cfg, "name": self.name})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class DebertaV3MultipleChoice(BaseModel):
    """DeBERTa-v3 multiple-choice head (context pooler + shared scorer)."""

    BASE_MODEL_CONFIG = BASE_MODEL_CONFIG
    BASE_WEIGHT_CONFIG = DEBERTA_V3_WEIGHT_CONFIG
    HF_MODEL_TYPE = "deberta-v2"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DebertaV2Model.config_from_hf(hf_config)

    @classmethod
    def from_release(cls, variant, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = DebertaV3Model.from_weights(variant, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    def __init__(
        self,
        num_choices=4,
        pooler_dropout=0.0,
        classifier_dropout=0.0,
        name="DebertaV3MultipleChoice",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "norm_rel_ebd": True,
            "conv_kernel_size": 0,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        input_ids = layers.Input(
            shape=(num_choices, None), dtype="int32", name="input_ids"
        )
        attention_mask = layers.Input(
            shape=(num_choices, None), dtype="int32", name="attention_mask"
        )
        token_type_ids = layers.Input(
            shape=(num_choices, None), dtype="int32", name="token_type_ids"
        )
        backbone = DebertaV3Model(**cfg, name=f"{name}_backbone")
        flatten = DebertaV2FlattenChoices(name="flatten_choices")
        seq = backbone(
            {
                "input_ids": flatten(input_ids),
                "attention_mask": flatten(attention_mask),
                "token_type_ids": flatten(token_type_ids),
            }
        )["last_hidden_state"]
        x = seq[:, 0]
        x = layers.Dropout(pooler_dropout)(x)
        x = layers.Dense(cfg["embed_dim"], name="pooler_dense")(x)
        x = layers.Activation("gelu", name="pooler_act")(x)
        x = layers.Dropout(classifier_dropout)(x)
        x = layers.Dense(1, name="classifier")(x)
        logits = DebertaV2UnflattenChoices(num_choices, name="unflatten_choices")(x)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        super().__init__(inputs=inputs, outputs=logits, name=name)
        self._cfg = cfg
        self.num_choices = num_choices
        self.pooler_dropout = pooler_dropout
        self.classifier_dropout = classifier_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                **self._cfg,
                "num_choices": self.num_choices,
                "pooler_dropout": self.pooler_dropout,
                "classifier_dropout": self.classifier_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
