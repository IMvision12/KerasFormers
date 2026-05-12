import keras
from keras import layers, ops

from kmodels.base import BaseModel
from kmodels.models.clip.clip_layers import (
    CLIPAttention,
    CLIPLogitScale,
    TextModelEmbedding,
    VisionModelEmbedding,
)

from .config import METACLIP2_CONFIG, METACLIP2_WEIGHTS
from .metaclip2_tokenizer import METACLIP2_EOS_TOKEN_ID


def quick_gelu(x):
    return x * ops.sigmoid(1.702 * x)


def _activation_layer(hidden_act):
    if hidden_act == "quick_gelu":
        return keras.layers.Lambda(quick_gelu)
    return keras.layers.Activation(hidden_act)


def residual_attention_block(
    x,
    proj_dim,
    num_heads,
    layer_name_prefix,
    layer_idx,
    causal_attention_mask=None,
    attention_mask=None,
    mlp_ratio=4.0,
    hidden_act="gelu",
):
    layer_prefix = f"{layer_name_prefix}_{layer_idx}"

    ln_1_output = keras.layers.LayerNormalization(
        epsilon=1e-5, name=f"{layer_prefix}_layernorm_1"
    )(x)

    mask = None
    if causal_attention_mask is not None:
        mask = ops.cast(causal_attention_mask, dtype=x.dtype)
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, dtype=x.dtype)
        mask = (
            ops.add(causal_attention_mask, attention_mask)
            if causal_attention_mask is not None
            else attention_mask
        )

    attention_output = CLIPAttention(
        proj_dim=proj_dim,
        num_heads=num_heads,
        name_prefix=f"{layer_prefix}_attn",
    )(ln_1_output, attention_mask=mask)[0]

    residual_1 = keras.layers.Add()([x, attention_output])
    ln_2_output = keras.layers.LayerNormalization(
        epsilon=1e-5, name=f"{layer_prefix}_layernorm_2"
    )(residual_1)

    mlp_intermediate_size = int(proj_dim * mlp_ratio)
    mlp_output = keras.layers.Dense(
        mlp_intermediate_size, name=f"{layer_prefix}_dense_1"
    )(ln_2_output)
    mlp_output = _activation_layer(hidden_act)(mlp_output)
    mlp_output = keras.layers.Dense(proj_dim, name=f"{layer_prefix}_dense_2")(
        mlp_output
    )

    output = keras.layers.Add()([residual_1, mlp_output])
    return output


def metaclip2_encoder(
    inputs,
    width,
    num_layers,
    heads,
    layer_prefix=None,
    causal_attention_mask=None,
    attention_mask=None,
    mlp_ratio=None,
    hidden_act="gelu",
):
    x = inputs
    for i in range(num_layers):
        x = residual_attention_block(
            x,
            proj_dim=width,
            num_heads=heads,
            layer_name_prefix=layer_prefix,
            layer_idx=i,
            causal_attention_mask=causal_attention_mask,
            attention_mask=attention_mask,
            mlp_ratio=mlp_ratio,
            hidden_act=hidden_act,
        )
    return x


def metaclip2_vision_features(
    inputs,
    input_resolution=224,
    patch_size=16,
    width=768,
    num_layers=12,
    heads=12,
    vision_mlp_ratio=4.0,
    hidden_act="gelu",
    data_format="channels_last",
):
    """MetaCLIP 2 vision encoder up through the transformer blocks.

    Returns the full token sequence ``(B, 1 + num_patches, width)`` (CLS
    + patch tokens) before any projection or pooling. Matches HF's
    ``MetaClip2VisionModel.last_hidden_state``.
    """
    patch_embeddings = keras.layers.Conv2D(
        filters=width,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="vision_model_conv",
    )(inputs)

    embeddings = VisionModelEmbedding(
        width, input_resolution, patch_size, data_format, name="vision_model_embeddings"
    )(patch_embeddings)

    x = keras.layers.LayerNormalization(epsilon=1e-5, name="vision_model_layernorm_1")(
        embeddings
    )
    return metaclip2_encoder(
        x,
        width=width,
        num_layers=num_layers,
        heads=heads,
        layer_prefix="vision_model_encoder",
        mlp_ratio=vision_mlp_ratio,
        hidden_act=hidden_act,
    )


def metaclip2_image_encoder(
    inputs,
    input_resolution=224,
    patch_size=16,
    width=768,
    num_layers=12,
    heads=12,
    output_dim=512,
    vision_mlp_ratio=4.0,
    hidden_act="gelu",
    data_format="channels_last",
):
    """Full MetaCLIP 2 vision encoder used by the contrastive head — features
    -> CLS token -> post-LN -> visual projection."""
    encoded = metaclip2_vision_features(
        inputs,
        input_resolution=input_resolution,
        patch_size=patch_size,
        width=width,
        num_layers=num_layers,
        heads=heads,
        vision_mlp_ratio=vision_mlp_ratio,
        hidden_act=hidden_act,
        data_format=data_format,
    )
    class_token = keras.layers.Lambda(lambda x: x[:, 0, :], name="extract_token")(
        encoded
    )
    x = keras.layers.LayerNormalization(epsilon=1e-5, name="vision_model_layernorm_2")(
        class_token
    )
    return keras.layers.Dense(output_dim, use_bias=False, name="visual_projection")(x)


def metaclip2_text_encoder(
    inputs,
    attention_mask,
    transformer_width,
    transformer_layers,
    transformer_heads,
    vocab_size,
    embed_dim,
    context_length,
    text_mlp_ratio,
    hidden_act="gelu",
    eos_token_id=METACLIP2_EOS_TOKEN_ID,
):
    x = TextModelEmbedding(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=transformer_width,
        name="text_model_embedding",
    )(inputs)

    causal_attention_mask = ops.triu(
        ops.ones((context_length, context_length)) * (-1e8), k=1
    )

    attention_mask_float = ops.cast(attention_mask, dtype="float32")
    expanded_mask = ops.reshape(attention_mask_float, (-1, 1, 1, context_length))
    expanded_mask = ops.repeat(expanded_mask, context_length, axis=2)
    expanded_mask = (1.0 - expanded_mask) * (-1e8)

    encoded_output = metaclip2_encoder(
        x,
        width=transformer_width,
        num_layers=transformer_layers,
        heads=transformer_heads,
        causal_attention_mask=causal_attention_mask,
        attention_mask=expanded_mask,
        mlp_ratio=text_mlp_ratio,
        hidden_act=hidden_act,
        layer_prefix="text_model_encoder",
    )

    layer_norm = keras.layers.LayerNormalization(name="text_model_layernorm")(
        encoded_output
    )

    eos_mask = ops.cast(ops.equal(inputs, eos_token_id), "int32")
    indices = ops.argmax(eos_mask, axis=-1)

    one_hot_indices = ops.one_hot(indices, context_length)
    selected_features = ops.einsum("bi,bij->bj", one_hot_indices, layer_norm)
    selected_features = ops.expand_dims(selected_features, axis=1)

    text_features = keras.layers.Dense(
        embed_dim, name="text_projection", use_bias=False
    )(selected_features)

    output = ops.squeeze(text_features, axis=1)
    return output


def metaclip2_head(image_embeddings, text_embeddings):
    normalize_image_features = ops.sqrt(
        ops.sum(ops.power(image_embeddings, 2), axis=-1, keepdims=True)
    )
    normalize_text_features = ops.sqrt(
        ops.sum(ops.power(text_embeddings, 2), axis=-1, keepdims=True)
    )
    image_embeddings = image_embeddings / normalize_image_features
    text_embeddings = text_embeddings / normalize_text_features
    logit_scale_layer = CLIPLogitScale(initial_value=0.07, name="logit_scale")
    image_logits, text_logits = logit_scale_layer([image_embeddings, text_embeddings])
    return image_logits, text_logits


def _metaclip2_resolve_image_shape(input_shape, image_resolution, data_format):
    if input_shape is not None:
        if data_format == "channels_first":
            if len(input_shape) == 3:
                channels = input_shape[0]
                image_size = min(input_shape[1], input_shape[2])
            else:
                channels = 3
                image_size = input_shape[0] if len(input_shape) >= 1 else 224
        else:
            if len(input_shape) >= 2:
                image_size = min(input_shape[0], input_shape[1])
            else:
                image_size = input_shape[0] if len(input_shape) >= 1 else 224
            channels = input_shape[2] if len(input_shape) == 3 else 3
    else:
        image_size, channels = image_resolution, 3
    return (
        [channels, image_size, image_size]
        if data_format == "channels_first"
        else [image_size, image_size, channels]
    ), image_size


@keras.saving.register_keras_serializable(package="kmodels")
class MetaClip2Model(BaseModel):
    """MetaCLIP 2 (multilingual / worldwide) contrastive vision-language model.

    Returns the projected vision + text embeddings (no head). Use
    :class:`MetaClip2ZeroShotClassify` for the contrastive similarity
    head or :class:`MetaClip2ImageClassify` for supervised image
    classification.

    MetaCLIP 2 is Meta's 2nd-generation CLIP, trained on multilingual data with
    the XLM-R tokenizer (vocab 901629). Architecturally it is identical to
    OpenAI CLIP except for:

    - Configurable MLP activation (``"gelu"`` or ``"quick_gelu"``).
    - EOS pooling uses explicit ``eos_token_id == 2`` match instead of
      argmax-over-token-ids (needed because mask_token_id > eos_token_id).
    - Wider / deeper text tower in larger variants.

    Reference:
      - https://arxiv.org/abs/2507.22062 ("MetaCLIP 2")
      - https://huggingface.co/docs/transformers/model_doc/metaclip_2
    """

    KMODELS_CONFIG = METACLIP2_CONFIG
    KMODELS_WEIGHTS = METACLIP2_WEIGHTS
    HF_MODEL_TYPE = "metaclip_2"

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config["vision_config"]
        tc = hf_config["text_config"]
        return {
            "embed_dim": hf_config["projection_dim"],
            "image_resolution": vc.get("image_size", 224),
            "vision_layers": vc["num_hidden_layers"],
            "vision_width": vc["hidden_size"],
            "vision_patch_size": vc["patch_size"],
            "vision_heads": vc.get("num_attention_heads"),
            "context_length": tc.get("max_position_embeddings", 77),
            "vocab_size": tc["vocab_size"],
            "transformer_width": tc["hidden_size"],
            "transformer_heads": tc["num_attention_heads"],
            "transformer_layers": tc["num_hidden_layers"],
            "vision_mlp_ratio": vc["intermediate_size"] / vc["hidden_size"],
            "text_mlp_ratio": tc["intermediate_size"] / tc["hidden_size"],
            "hidden_act": vc.get("hidden_act", "gelu"),
            "eos_token_id": tc.get("eos_token_id", METACLIP2_EOS_TOKEN_ID),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kmodels.models.metaclip2.convert_metaclip2_hf_to_keras import (
            transfer_metaclip2_weights,
        )

        transfer_metaclip2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        vision_heads=None,
        context_length=77,
        vocab_size=901629,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        vision_mlp_ratio=4.0,
        text_mlp_ratio=4.0,
        hidden_act="gelu",
        eos_token_id=METACLIP2_EOS_TOKEN_ID,
        input_shape=None,
        input_tensor=None,
        name="MetaClip2Model",
        **kwargs,
    ):
        if vision_heads is None:
            vision_heads = vision_width // 64
        data_format = keras.backend.image_data_format()

        if input_shape is not None:
            if data_format == "channels_first":
                if len(input_shape) == 3:
                    channels = input_shape[0]
                    image_size = min(input_shape[1], input_shape[2])
                else:
                    channels = 3
                    image_size = input_shape[0] if len(input_shape) >= 1 else 224
            else:
                if len(input_shape) >= 2:
                    image_size = min(input_shape[0], input_shape[1])
                else:
                    image_size = input_shape[0] if len(input_shape) >= 1 else 224
                channels = input_shape[2] if len(input_shape) == 3 else 3
        else:
            image_size = image_resolution
            channels = 3

        if data_format == "channels_first":
            image_input_shape = [channels, image_size, image_size]
        else:
            image_input_shape = [image_size, image_size, channels]

        if isinstance(input_tensor, dict):
            images_input = input_tensor.get("images") or layers.Input(
                shape=image_input_shape, name="images"
            )
            token_ids_input = input_tensor.get("token_ids") or layers.Input(
                shape=[context_length], name="token_ids"
            )
            padding_mask_input = input_tensor.get("padding_mask") or layers.Input(
                shape=[context_length], name="padding_mask"
            )
        else:
            images_input = layers.Input(shape=image_input_shape, name="images")
            token_ids_input = layers.Input(shape=[context_length], name="token_ids")
            padding_mask_input = layers.Input(
                shape=[context_length], name="padding_mask"
            )

        image_embeddings = metaclip2_image_encoder(
            images_input,
            input_resolution=image_size,
            patch_size=vision_patch_size,
            width=vision_width,
            num_layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
            data_format=data_format,
        )

        text_embeddings = metaclip2_text_encoder(
            token_ids_input,
            attention_mask=padding_mask_input,
            transformer_width=transformer_width,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            text_mlp_ratio=text_mlp_ratio,
            context_length=context_length,
            hidden_act=hidden_act,
            eos_token_id=eos_token_id,
        )

        outputs = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }
        inputs = {
            "images": images_input,
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.embed_dim = embed_dim
        self.image_resolution = image_resolution
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.vision_heads = vision_heads
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.vision_mlp_ratio = vision_mlp_ratio
        self.text_mlp_ratio = text_mlp_ratio
        self.hidden_act = hidden_act
        self.eos_token_id = eos_token_id
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        image_shape_with_batch = self.input_shape[0]
        if image_shape_with_batch[0] is None:
            image_input_shape = image_shape_with_batch[1:]
        else:
            image_input_shape = image_shape_with_batch
        config.update(
            {
                "embed_dim": self.embed_dim,
                "image_resolution": self.image_resolution,
                "input_shape": image_input_shape,
                "vision_layers": self.vision_layers,
                "vision_width": self.vision_width,
                "vision_patch_size": self.vision_patch_size,
                "vision_heads": self.vision_heads,
                "context_length": self.context_length,
                "vocab_size": self.vocab_size,
                "transformer_width": self.transformer_width,
                "transformer_heads": self.transformer_heads,
                "transformer_layers": self.transformer_layers,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "text_mlp_ratio": self.text_mlp_ratio,
                "hidden_act": self.hidden_act,
                "eos_token_id": self.eos_token_id,
                "input_tensor": self.input_tensor,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class MetaClip2ZeroShotClassify(BaseModel):
    """MetaCLIP 2 + contrastive similarity head for zero-shot classification / retrieval.

    Composes the same vision + text encoders as :class:`MetaClip2Model`
    and adds the standard CLIP-style head — L2-normalize both sides,
    then a learnable ``logit_scale`` temperature on the cosine-similarity
    matrix. Output is the ``(B, B)`` image-vs-text similarity logits.

    >>> MetaClip2ZeroShotClassify.from_weights("metaclip2_worldwide_b32_224")
    >>> MetaClip2ZeroShotClassify.from_weights("hf:facebook/metaclip-2-worldwide-b32")
    """

    KMODELS_CONFIG = METACLIP2_CONFIG
    KMODELS_WEIGHTS = METACLIP2_WEIGHTS
    HF_MODEL_TYPE = "metaclip_2"

    @classmethod
    def config_from_hf(cls, hf_config):
        return MetaClip2Model.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kmodels.models.metaclip2.convert_metaclip2_hf_to_keras import (
            transfer_metaclip2_weights,
        )

        transfer_metaclip2_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        vision_heads=None,
        context_length=77,
        vocab_size=901629,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        vision_mlp_ratio=4.0,
        text_mlp_ratio=4.0,
        hidden_act="gelu",
        eos_token_id=METACLIP2_EOS_TOKEN_ID,
        input_shape=None,
        input_tensor=None,
        name="MetaClip2ZeroShotClassify",
        **kwargs,
    ):
        base = MetaClip2Model(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            vision_heads=vision_heads,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            vision_mlp_ratio=vision_mlp_ratio,
            text_mlp_ratio=text_mlp_ratio,
            hidden_act=hidden_act,
            eos_token_id=eos_token_id,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_base",
        )
        image_logits, text_logits = metaclip2_head(
            base.output["image_embeddings"], base.output["text_embeddings"]
        )

        super().__init__(
            inputs=base.input,
            outputs={"image_logits": image_logits, "text_logits": text_logits},
            name=name,
            **kwargs,
        )

        self.embed_dim = embed_dim
        self.image_resolution = image_resolution
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.vision_heads = (
            vision_heads if vision_heads is not None else vision_width // 64
        )
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.vision_mlp_ratio = vision_mlp_ratio
        self.text_mlp_ratio = text_mlp_ratio
        self.hidden_act = hidden_act
        self.eos_token_id = eos_token_id
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        image_shape_with_batch = self.input_shape["images"]
        if image_shape_with_batch[0] is None:
            image_input_shape = image_shape_with_batch[1:]
        else:
            image_input_shape = image_shape_with_batch
        config.update(
            {
                "embed_dim": self.embed_dim,
                "image_resolution": self.image_resolution,
                "input_shape": image_input_shape,
                "vision_layers": self.vision_layers,
                "vision_width": self.vision_width,
                "vision_patch_size": self.vision_patch_size,
                "vision_heads": self.vision_heads,
                "context_length": self.context_length,
                "vocab_size": self.vocab_size,
                "transformer_width": self.transformer_width,
                "transformer_heads": self.transformer_heads,
                "transformer_layers": self.transformer_layers,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "text_mlp_ratio": self.text_mlp_ratio,
                "hidden_act": self.hidden_act,
                "eos_token_id": self.eos_token_id,
                "input_tensor": self.input_tensor,
                "name": self.name,
                "trainable": self.trainable,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kmodels")
class MetaClip2ImageClassify(BaseModel):
    """MetaCLIP 2 vision encoder + linear image-classification head.

    Mirrors HF's ``MetaClip2ForImageClassification``: uses **only the
    vision encoder** (no text encoder, no visual projection, no post-LN,
    no ``logit_scale``), drops the CLS token, mean-pools the patch
    tokens, and applies a single linear classifier producing
    ``num_labels`` logits.
    """

    KMODELS_CONFIG = METACLIP2_CONFIG
    KMODELS_WEIGHTS = METACLIP2_WEIGHTS
    HF_MODEL_TYPE = "metaclip_2"

    _RELEASE_CONFIG_KEYS = (
        "image_resolution",
        "vision_layers",
        "vision_width",
        "vision_patch_size",
        "vision_heads",
        "vision_mlp_ratio",
        "hidden_act",
    )

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        if variant not in cls.KMODELS_CONFIG:
            available = sorted(cls.KMODELS_CONFIG.keys())
            raise ValueError(
                f"Unknown variant '{variant}' for {cls.__name__}. "
                f"Available variants: {available}"
            )
        full = cls.KMODELS_CONFIG[variant]
        config = {k: v for k, v in full.items() if k in cls._RELEASE_CONFIG_KEYS}
        config.update(kwargs)
        model = cls(**config)

        if load_weights:
            src = MetaClip2Model.from_weights(variant)

            def _key(w):
                return "/".join(w.path.split("/")[-2:])

            src_map = {_key(w): w for w in src.weights}
            for dst_w in model.weights:
                src_w = src_map.get(_key(dst_w))
                if src_w is not None and tuple(src_w.shape) == tuple(dst_w.shape):
                    dst_w.assign(src_w)
            del src

        return model

    @classmethod
    def config_from_hf(cls, hf_config):
        from kmodels.base.base_model import hf_num_labels

        vc = hf_config["vision_config"]
        config = {
            "image_resolution": vc.get("image_size", 224),
            "vision_layers": vc["num_hidden_layers"],
            "vision_width": vc["hidden_size"],
            "vision_patch_size": vc["patch_size"],
            "vision_mlp_ratio": vc["intermediate_size"] / vc["hidden_size"],
            "hidden_act": vc.get("hidden_act", "gelu"),
        }
        try:
            config["num_labels"] = hf_num_labels(hf_config)
        except KeyError:
            config["num_labels"] = 1000
        return config

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kmodels.models.metaclip2.convert_metaclip2_hf_to_keras import (
            transfer_metaclip2_image_classify_weights,
        )

        transfer_metaclip2_image_classify_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        num_labels=1000,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        vision_heads=None,
        vision_mlp_ratio=4.0,
        hidden_act="gelu",
        input_shape=None,
        input_tensor=None,
        name="MetaClip2ImageClassify",
        **kwargs,
    ):
        if vision_heads is None:
            vision_heads = vision_width // 64
        data_format = keras.backend.image_data_format()
        image_input_shape, image_size = _metaclip2_resolve_image_shape(
            input_shape, image_resolution, data_format
        )

        if input_tensor is None:
            images_input = layers.Input(shape=image_input_shape, name="images")
        else:
            images_input = input_tensor

        encoded = metaclip2_vision_features(
            images_input,
            input_resolution=image_size,
            patch_size=vision_patch_size,
            width=vision_width,
            num_layers=vision_layers,
            heads=vision_heads,
            vision_mlp_ratio=vision_mlp_ratio,
            hidden_act=hidden_act,
            data_format=data_format,
        )

        patches = layers.Lambda(lambda t: t[:, 1:, :], name="drop_cls")(encoded)
        pooled = layers.GlobalAveragePooling1D(name="patch_pool")(patches)
        logits = layers.Dense(num_labels, name="classifier")(pooled)

        super().__init__(inputs=images_input, outputs=logits, name=name, **kwargs)

        self.num_labels = num_labels
        self.image_resolution = image_size
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.vision_heads = vision_heads
        self.vision_mlp_ratio = vision_mlp_ratio
        self.hidden_act = hidden_act
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        image_shape_with_batch = self.input_shape
        if image_shape_with_batch[0] is None:
            image_input_shape = image_shape_with_batch[1:]
        else:
            image_input_shape = image_shape_with_batch
        config.update(
            {
                "num_labels": self.num_labels,
                "image_resolution": self.image_resolution,
                "input_shape": image_input_shape,
                "vision_layers": self.vision_layers,
                "vision_width": self.vision_width,
                "vision_patch_size": self.vision_patch_size,
                "vision_heads": self.vision_heads,
                "vision_mlp_ratio": self.vision_mlp_ratio,
                "hidden_act": self.hidden_act,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
