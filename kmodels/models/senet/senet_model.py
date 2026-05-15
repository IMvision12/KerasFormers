import keras
from keras import layers

from kmodels.models.resnet.resnet_model import (
    ResNetClassify,
    ResNetModel,
    bottleneck_block,
)
from kmodels.models.resnext.resnext_model import resnext_block
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import SENET_MODEL_CONFIG, SENET_WEIGHT_CONFIG

_BLOCK_FN_LOOKUP = {
    "bottleneck_block": bottleneck_block,
    "resnext_block": resnext_block,
}


def resolve_block_fn(kwargs):
    """Resolve a ``block_fn_name`` string in ``kwargs`` to the actual callable.

    Subclass ``__init__`` methods call this to support both SE-ResNet
    (``bottleneck_block``) and SE-ResNeXt (``resnext_block``) variants
    from a single shared config dict.

    Args:
        kwargs: Mutable dict of keyword arguments. If it contains a
            ``"block_fn_name"`` entry, that key is popped and a matching
            ``"block_fn"`` callable from ``_BLOCK_FN_LOOKUP`` is inserted.

    Returns:
        None. ``kwargs`` is mutated in place.
    """
    name = kwargs.pop("block_fn_name", None)
    if name is not None:
        kwargs["block_fn"] = _BLOCK_FN_LOOKUP[name]


@keras.saving.register_keras_serializable(package="kmodels")
class SENetModel(ResNetModel):
    """Instantiates the Squeeze-and-Excitation Network (SENet) backbone.

    SENet augments a ResNet or ResNeXt trunk with Squeeze-and-Excitation
    blocks — per-channel attention weights computed by global average
    pooling followed by a 2-layer MLP (reduce -> expand) and a sigmoid
    gate that rescales each channel of the residual branch. The output
    tensor is the last layer output before the classifier head — the
    final-stage feature map ``(B, H, W, C)``, unpooled and head-free.
    :class:`SENetClassify` composes this model and applies a
    GlobalAveragePooling2D + Dense head to produce logits.

    References:
    - [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

    Args:
        senet: Boolean, whether to apply Squeeze-and-Excitation inside
            each block. Defaults to `True`.
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps. Defaults to `False`.
        name: String, the name of the model. Defaults to `"SENetModel"`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`ResNetModel`, including ``block_fn`` /
            ``block_fn_name`` (selects bottleneck vs. ResNeXt block),
            ``block_repeats``, ``filters``, ``groups``, ``width_factor``,
            ``include_normalization``, ``normalization_mode``,
            ``input_shape``, and ``input_tensor``.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = {
        variant: SENET_MODEL_CONFIG[meta["model"]]
        for variant, meta in SENET_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = SENET_WEIGHT_CONFIG

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = SENetClassify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    def __init__(self, senet=True, as_backbone=False, name="SENetModel", **kwargs):
        resolve_block_fn(kwargs)
        super().__init__(senet=senet, as_backbone=as_backbone, name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class SENetClassify(ResNetClassify):
    """Instantiates the Squeeze-and-Excitation Network (SENet) classifier.

    This classifier wraps a :class:`SENetModel` backbone and attaches a
    GlobalAveragePooling2D + Dense head to produce ``num_classes`` class
    logits. All architectural parameters are forwarded to the underlying
    :class:`SENetModel`; only ``num_classes`` and
    ``classifier_activation`` are head-specific. Both ``seresnet*``
    (bottleneck block) and ``seresnext*`` (grouped block) variants are
    supported — the block function is selected per-variant via the
    ``block_fn_name`` key in :data:`SENET_MODEL_CONFIG`.

    References:
    - [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

    Args:
        block_fn: Callable, the residual block builder. Should accept
            ``(x, filters, strides=1, downsample=False, block_name=None)``
            and additional keyword arguments (``groups``,
            ``width_factor`` for the ResNeXt block). May be overridden
            by ``block_fn_name`` in ``kwargs``.
            Defaults to `bottleneck_block`.
        block_repeats: List of ints, number of residual blocks per stage.
            Defaults to `[2, 2, 2, 2]`.
        filters: List of ints, base filter counts per stage (the final
            output width is ``filters[i] * expansion``).
            Defaults to `[64, 128, 256, 512]`.
        groups: Integer, number of groups for grouped convolution (used
            when ``block_fn`` is the ResNeXt block). Defaults to `32`.
        senet: Boolean, whether to apply Squeeze-and-Excitation inside
            each block. Defaults to `True`.
        width_factor: Integer, width scaling factor for the grouped
            convolution channels. Defaults to `2`.
        include_normalization: Boolean, whether to prepend an
            :class:`~kmodels.layers.ImageNormalizationLayer` at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'` (default), `'inception'`,
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_shape: Optional tuple specifying the shape of the input
            data. If `None`, derived from the active Keras data format
            with a default size of 224. Defaults to `None`.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        num_classes: Integer, the number of output classes for
            classification. Defaults to `1000`.
        classifier_activation: String or callable, activation function
            for the final Dense layer. Use `"linear"` to return raw
            logits or `"softmax"` to return class probabilities.
            Defaults to `"linear"`.
        name: String, the name of the model. The internal backbone is
            named `f"{name}_backbone"`. Defaults to `"SENetClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_MODEL_CONFIG = {
        variant: SENET_MODEL_CONFIG[meta["model"]]
        for variant, meta in SENET_WEIGHT_CONFIG.items()
    }
    BASE_WEIGHT_CONFIG = SENET_WEIGHT_CONFIG

    def __init__(
        self,
        block_fn=bottleneck_block,
        block_repeats=[2, 2, 2, 2],
        filters=[64, 128, 256, 512],
        groups=32,
        senet=True,
        width_factor=2,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="SENetClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)
        resolve_block_fn(kwargs)
        # If `block_fn_name` was provided, it has overwritten kwargs["block_fn"].
        block_fn = kwargs.pop("block_fn", block_fn)

        data_format = keras.config.image_data_format()

        backbone = SENetModel(
            block_fn=block_fn,
            block_repeats=block_repeats,
            filters=filters,
            groups=groups,
            senet=senet,
            width_factor=width_factor,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            kernel_initializer="zeros",
            name="predictions",
        )(x)

        super(ResNetClassify, self).__init__(
            inputs=backbone.input, outputs=out, name=name, **kwargs
        )

        self.block_fn = block_fn
        self.block_repeats = block_repeats
        self.filters = filters
        self.groups = groups
        self.senet = senet
        self.width_factor = width_factor
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
