"""timm SENet -> Keras weight transfer.

Re-uses :func:`transfer_resnet_weights` (the timm naming is identical
across resnet / resnext / senet families).
"""

from kmodels.models.resnet.convert_resnet_torch_to_keras import (
    transfer_resnet_weights as transfer_senet_weights,  # noqa: F401
)

if __name__ == "__main__":
    import gc

    import keras

    from kmodels.base.base_model import load_hf_state_dict
    from kmodels.models.senet import SENet
    from kmodels.models.senet.config import SENET_CONFIG

    for variant, cfg in SENET_CONFIG.items():
        timm_id = cfg["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = load_hf_state_dict(f"timm/{timm_id}")
        keras_model = SENet.from_weights(variant, load_weights=False)
        transfer_senet_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state
        keras.backend.clear_session()
        gc.collect()
