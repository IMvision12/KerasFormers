import gc

import keras
import timm

from kerasformers.base.base_model import download_hf_state_dict
from kerasformers.models.resnet.convert_resnet_timm_to_keras import (
    transfer_resnet_weights as transfer_senet_weights,
)
from kerasformers.models.senet import SENetImageClassify
from kerasformers.models.senet.config import SENET_WEIGHT_CONFIG
from kerasformers.weight_utils import verify_cls_model_equivalence

if __name__ == "__main__":
    for variant, meta in SENET_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = SENetImageClassify.from_weights(
            variant, load_weights=False, include_normalization=False
        )
        transfer_senet_weights(keras_model, state)

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="torch_to_keras",
            run_performance=False,
            atol=1e-4,
            rtol=1e-4,
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
