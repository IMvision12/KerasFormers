"""SigLIP 2 model classes.

SigLIP 2 shares the SigLIP v1 architecture (vision + text dual encoder
with attention pooling on the vision side, last-token pooling + Dense
on the text side, sigmoid-similarity head). The differences live in
preprocessing — Gemma SentencePiece tokenizer, vocab 256000, and a
different set of pretrained checkpoints.

These three classes are thin wrappers over the siglip module's encoder
code with SigLIP 2's variant registry. ``HF_MODEL_TYPE = "siglip2"``
keeps each class scoped to its HuggingFace ``model_type``.
"""

import keras

from kerasformers.base import BaseModel
from kerasformers.models.siglip.siglip_model import (
    SigLIPImageClassify,
    SigLIPModel,
    siglip_head,
)

from .config import SIGLIP2_CONFIG, SIGLIP2_WEIGHTS


@keras.saving.register_keras_serializable(package="kerasformers")
class SigLIP2Model(SigLIPModel):
    """SigLIP 2 dual encoder (no contrastive head).

    Architecture identical to :class:`SigLIPModel`; differs only in the
    variant registry and ``HF_MODEL_TYPE``. Returns the projected vision
    + text embeddings.

    >>> SigLIP2Model.from_weights("siglip2_base_p16_224")
    >>> SigLIP2Model.from_weights("hf:google/siglip2-base-patch16-224")
    """

    BASE_MODEL_CONFIG = SIGLIP2_CONFIG
    BASE_WEIGHT_CONFIG = SIGLIP2_WEIGHTS

    HF_MODEL_TYPE = "siglip"

    def __init__(self, *args, name="SigLIP2Model", **kwargs):
        super().__init__(*args, name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kerasformers")
class SigLIP2ZeroShotClassify(BaseModel):
    """SigLIP 2 + sigmoid-similarity head for zero-shot classification.

    Composes :class:`SigLIP2Model` + the standard SigLIP head (L2-normalize
    both sides, learnable ``logit_scale`` and ``logit_bias`` on the
    cosine-similarity matrix).
    """

    BASE_MODEL_CONFIG = SIGLIP2_CONFIG
    BASE_WEIGHT_CONFIG = SIGLIP2_WEIGHTS

    HF_MODEL_TYPE = "siglip"

    @classmethod
    def config_from_hf(cls, hf_config):
        return SigLIP2Model.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.siglip.convert_siglip_torch_to_keras import (
            transfer_siglip_weights,
        )

        transfer_siglip_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        input_image_shape=224,
        patch_size=16,
        vision_hidden_dim=768,
        vision_num_layers=12,
        vision_num_heads=12,
        vision_intermediate_dim=3072,
        vocabulary_size=256000,
        embed_dim=768,
        text_hidden_dim=768,
        text_num_layers=12,
        text_num_heads=12,
        text_intermediate_dim=3072,
        max_sequence_length=64,
        input_tensor=None,
        name="SigLIP2ZeroShotClassify",
        **kwargs,
    ):
        base = SigLIP2Model(
            input_image_shape=input_image_shape,
            patch_size=patch_size,
            vision_hidden_dim=vision_hidden_dim,
            vision_num_layers=vision_num_layers,
            vision_num_heads=vision_num_heads,
            vision_intermediate_dim=vision_intermediate_dim,
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            text_hidden_dim=text_hidden_dim,
            text_num_layers=text_num_layers,
            text_num_heads=text_num_heads,
            text_intermediate_dim=text_intermediate_dim,
            max_sequence_length=max_sequence_length,
            input_tensor=input_tensor,
            name=f"{name}_base",
        )
        image_logits, text_logits = siglip_head(
            base.output["image_embeddings"], base.output["text_embeddings"]
        )

        super().__init__(
            inputs=base.input,
            outputs={"image_logits": image_logits, "text_logits": text_logits},
            name=name,
            **kwargs,
        )

        self.input_image_shape = base.input_image_shape
        self.patch_size = patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_intermediate_dim = vision_intermediate_dim
        self.vocabulary_size = vocabulary_size
        self.embed_dim = embed_dim
        self.text_hidden_dim = text_hidden_dim
        self.text_num_layers = text_num_layers
        self.text_num_heads = text_num_heads
        self.text_intermediate_dim = text_intermediate_dim
        self.max_sequence_length = max_sequence_length
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_image_shape": self.input_image_shape,
                "patch_size": self.patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "vision_intermediate_dim": self.vision_intermediate_dim,
                "vocabulary_size": self.vocabulary_size,
                "embed_dim": self.embed_dim,
                "text_hidden_dim": self.text_hidden_dim,
                "text_num_layers": self.text_num_layers,
                "text_num_heads": self.text_num_heads,
                "text_intermediate_dim": self.text_intermediate_dim,
                "max_sequence_length": self.max_sequence_length,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class SigLIP2ImageClassify(SigLIPImageClassify):
    """SigLIP 2 vision encoder + linear classifier head.

    Mirrors :class:`SigLIPImageClassify`; the only differences are the
    variant registry (``SIGLIP2_CONFIG`` / ``SIGLIP2_WEIGHTS``) and that
    ``from_release`` warm-starts the encoder from a
    :class:`SigLIP2Model` checkpoint.
    """

    BASE_MODEL_CONFIG = SIGLIP2_CONFIG
    BASE_WEIGHT_CONFIG = SIGLIP2_WEIGHTS

    HF_MODEL_TYPE = "siglip"

    @classmethod
    def _release_warm_start_cls(cls):
        return SigLIP2Model
