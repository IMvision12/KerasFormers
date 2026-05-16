"""timm FlexiViT -> Keras weight transfer (shares ViT mapping)."""

from kerasformers.models.vit.convert_vit_torch_to_keras import (
    transfer_vit_weights as transfer_flexivit_weights,
)

__all__ = ["transfer_flexivit_weights"]


if __name__ == "__main__":
    import gc

    import keras

    from kerasformers.base.base_model import download_hf_state_dict
    from kerasformers.models.flexivit import FlexiViTClassify
    from kerasformers.models.flexivit.config import FLEXIVIT_WEIGHT_CONFIG

    for variant, meta in FLEXIVIT_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = FlexiViTClassify.from_weights(variant, load_weights=False)
        transfer_flexivit_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
