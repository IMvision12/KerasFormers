import keras
from keras import layers

from kmodels.models.resnet.resnet_model import (
    ResNetClassify,
    ResNetModel,
    bottleneck_block,
)
from kmodels.models.resnext.resnext_model import resnext_block
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import SENET_CONFIG, SENET_WEIGHTS

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
    """SE-ResNet / SE-ResNeXt trunk returning the final stage feature map."""

    KMODELS_CONFIG = SENET_CONFIG
    KMODELS_WEIGHTS = SENET_WEIGHTS

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
    """Squeeze-and-Excitation ResNet / ResNeXt classifier.

    Composes a :class:`SENetModel` backbone with the GAP + Dense head.
    Covers both ``seresnet*`` (bottleneck block) and ``seresnext*``
    (grouped block) variants — block_fn is selected per-variant via the
    ``block_fn_name`` key in :data:`SENET_CONFIG`.

    >>> SENetClassify.from_weights("seresnet50_a1_in1k")
    >>> SENetClassify.from_weights("seresnext50_32x4d_racm_in1k")
    >>> SENetClassify.from_weights("timm:timm/seresnet50.a1_in1k")
    """

    KMODELS_CONFIG = SENET_CONFIG
    KMODELS_WEIGHTS = SENET_WEIGHTS

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
