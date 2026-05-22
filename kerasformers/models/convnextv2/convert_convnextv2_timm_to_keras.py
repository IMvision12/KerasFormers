import gc

import keras
import numpy as np
import timm

from kerasformers.base.base_model import download_hf_state_dict
from kerasformers.models.convnext.convert_convnext_timm_to_keras import (
    transfer_convnext_weights as transfer_convnextv2_weights,
)
from kerasformers.models.convnextv2 import ConvNeXtV2ImageClassify
from kerasformers.models.convnextv2.config import CONVNEXTV2_WEIGHT_CONFIG
from kerasformers.weight_utils import verify_cls_model_equivalence

__all__ = ["transfer_convnextv2_weights"]


if __name__ == "__main__":
    for variant, meta in CONVNEXTV2_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ConvNeXtV2ImageClassify.from_weights(
            variant, load_weights=False, include_normalization=False
        )
        transfer_convnextv2_weights(keras_model, state)

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

        total_params = sum(int(np.prod(w.shape)) for w in keras_model.weights)
        total_gb = (total_params * 4) / (1024**3)
        if total_gb > 1.7:
            out_path = f"{variant}.weights.json"
            keras_model.save_weights(out_path, max_shard_size=1.7)
            print(f"  Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
        else:
            out_path = f"{variant}.weights.h5"
            keras_model.save_weights(out_path)
            print(f"  Saved -> {out_path} (~{total_gb:.2f} GB)")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
