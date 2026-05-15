"""ConvNeXtV2 as thin :class:`ConvNeXt` subclasses (timm-ported)."""

import keras
from keras import layers

from kmodels.models.convnext.convert_convnext_torch_to_keras import (
    transfer_convnext_weights,
)
from kmodels.models.convnext.convnext_model import (
    ConvNeXtClassify,
    ConvNeXtModel,
)
from kmodels.weight_utils import copy_weights_by_path_suffix

from .config import CONVNEXTV2_MODEL_CONFIG, CONVNEXTV2_WEIGHT_CONFIG


@keras.saving.register_keras_serializable(package="kmodels")
class ConvNeXtV2Model(ConvNeXtModel):
    """ConvNeXtV2 backbone returning the final stage feature map ``(B, H, W, C)``."""

    KMODELS_CONFIG = CONVNEXTV2_MODEL_CONFIG
    KMODELS_WEIGHTS = CONVNEXTV2_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def from_release(cls, variant, load_weights=True, **kwargs):
        model = super().from_release(variant, load_weights=False, **kwargs)
        if load_weights:
            src = ConvNeXtV2Classify.from_weights(variant)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_convnext_weights(keras_model, state_dict)

    def __init__(self, as_backbone=False, name="ConvNeXtV2Model", **kwargs):
        super().__init__(as_backbone=as_backbone, name=name, **kwargs)


@keras.saving.register_keras_serializable(package="kmodels")
class ConvNeXtV2Classify(ConvNeXtClassify):
    """ConvNeXtV2 classifier (GRN + post-FCMAE finetune).

    Reference:
    - [ConvNeXt V2](https://arxiv.org/abs/2301.00808)

    Construction:

    >>> ConvNeXtV2Classify.from_weights("convnextv2_base_fcmae_ft_in22k_in1k")
    >>> ConvNeXtV2Classify.from_weights("timm:timm/convnextv2_base.fcmae_ft_in22k_in1k")
    """

    KMODELS_CONFIG = CONVNEXTV2_MODEL_CONFIG
    KMODELS_WEIGHTS = CONVNEXTV2_WEIGHT_CONFIG
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        transfer_convnext_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        projection_dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        use_conv=False,
        use_grn=False,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_shape=None,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="ConvNeXtV2Classify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = ConvNeXtV2Model(
            depths=depths,
            projection_dims=projection_dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            use_conv=use_conv,
            use_grn=use_grn,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_shape=input_shape,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        x = layers.LayerNormalization(axis=-1, epsilon=1e-6, name="final_layernorm")(x)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        # Skip ConvNeXtClassify.__init__; go straight to BaseModel.
        super(ConvNeXtClassify, self).__init__(
            inputs=backbone.input, outputs=out, name=name, **kwargs
        )

        self.depths = list(depths)
        self.projection_dims = list(projection_dims)
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.use_conv = use_conv
        self.use_grn = use_grn
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
