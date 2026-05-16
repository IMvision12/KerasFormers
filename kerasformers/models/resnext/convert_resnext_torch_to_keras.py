"""timm ResNeXtClassify -> Keras weight transfer.

The timm-name -> keras-name mapping is identical to ResNet's, so this
script just re-uses :func:`transfer_resnet_weights`. The ``__main__``
block iterates :data:`RESNEXT_MODEL_CONFIG` and saves one ``.weights.h5`` per
variant.
"""

from kerasformers.models.resnet.convert_resnet_torch_to_keras import (
    transfer_resnet_weights as transfer_resnext_weights,  # noqa: F401
)

if __name__ == "__main__":
    import gc

    import keras

    from kerasformers.base.base_model import download_hf_state_dict
    from kerasformers.models.resnext import ResNeXtClassify
    from kerasformers.models.resnext.config import RESNEXT_WEIGHT_CONFIG

    for variant, meta in RESNEXT_WEIGHT_CONFIG.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ResNeXtClassify.from_weights(variant, load_weights=False)
        transfer_resnext_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
