"""timm FlexiViT -> Keras weight transfer (shares ViT mapping)."""

import gc

import keras
import timm

from kerasformers.base.base_model import download_hf_state_dict
from kerasformers.models.flexivit import FlexiViTImageClassify
from kerasformers.models.flexivit.config import FLEXIVIT_WEIGHT_CONFIG
from kerasformers.models.vit.convert_vit_torch_to_keras import (
    transfer_vit_weights as transfer_flexivit_weights,
)
from kerasformers.weight_utils import verify_cls_model_equivalence

__all__ = ["transfer_flexivit_weights"]


if __name__ == "__main__":
    for variant, meta in FLEXIVIT_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = FlexiViTImageClassify.from_weights(variant, load_weights=False)
        transfer_flexivit_weights(keras_model, state)

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="torch_to_keras",
            run_performance=False,
        )
        if not results["standard_input"]:
            raise ValueError(
                "Model equivalence test failed - model outputs do not match for standard input"
            )

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
