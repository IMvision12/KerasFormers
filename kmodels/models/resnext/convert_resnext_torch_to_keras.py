"""timm ResNeXt -> Keras weight transfer.

The timm-name -> keras-name mapping is identical to ResNet's, so this
script just re-uses :func:`transfer_resnet_weights`. The ``__main__``
block iterates :data:`RESNEXT_CONFIG` and saves one ``.weights.h5`` per
variant.
"""

from kmodels.models.resnet.convert_resnet_torch_to_keras import (
    transfer_resnet_weights as transfer_resnext_weights,  # noqa: F401
)

if __name__ == "__main__":
    import gc

    import keras

    from kmodels.base.base_model import load_hf_state_dict
    from kmodels.models.resnext import ResNeXt
    from kmodels.models.resnext.config import RESNEXT_CONFIG

    for variant, cfg in RESNEXT_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = load_hf_state_dict(f"timm/{timm_id}")
        keras_model = ResNeXt.from_weights(variant, load_weights=False)
        transfer_resnext_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
