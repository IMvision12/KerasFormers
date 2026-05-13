"""timm ConvNeXtV2 -> Keras weight transfer (shares ConvNeXt mapping)."""

from kmodels.models.convnext.convert_convnext_torch_to_keras import (
    transfer_convnext_weights as transfer_convnextv2_weights,
)

__all__ = ["transfer_convnextv2_weights"]


if __name__ == "__main__":
    import gc

    import keras

    from kmodels.base.base_model import download_hf_state_dict
    from kmodels.models.convnextv2 import ConvNeXtV2
    from kmodels.models.convnextv2.config import CONVNEXTV2_CONFIG

    for variant, cfg in CONVNEXTV2_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ConvNeXtV2.from_weights(variant, load_weights=False)
        transfer_convnextv2_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
