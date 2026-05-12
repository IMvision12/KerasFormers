"""HuggingFace SigLIP 2 -> Keras weight transfer.

SigLIP 2 shares the SigLIP v1 architecture, so the actual transfer is
:func:`kmodels.models.siglip.convert_siglip_torch_to_keras.transfer_siglip_weights`.
This module exists as a thin entrypoint that wires the SigLIP 2 variant
ids to their HuggingFace repos and runs the conversion through
:class:`SigLIP2ZeroShotClassify` (so the saved ``.weights.h5`` includes
the contrastive head).
"""

from typing import Dict

import numpy as np

from kmodels.models.siglip.convert_siglip_torch_to_keras import transfer_siglip_weights


def transfer_siglip2_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    """Alias for :func:`transfer_siglip_weights` — kept for API symmetry
    with the rest of the SigLIP 2 module."""
    transfer_siglip_weights(keras_model, hf_state_dict)


if __name__ == "__main__":
    import gc

    import keras
    from transformers import SiglipModel

    from kmodels.models.siglip2 import SigLIP2ZeroShotClassify

    SIGLIP2_CONVERSION_CONFIG = [
        ("siglip2_base_p16_224", "google/siglip2-base-patch16-224"),
        ("siglip2_base_p16_256", "google/siglip2-base-patch16-256"),
        ("siglip2_base_p16_384", "google/siglip2-base-patch16-384"),
        ("siglip2_base_p16_512", "google/siglip2-base-patch16-512"),
        ("siglip2_base_p32_256", "google/siglip2-base-patch32-256"),
        ("siglip2_large_p16_256", "google/siglip2-large-patch16-256"),
        ("siglip2_large_p16_384", "google/siglip2-large-patch16-384"),
        ("siglip2_large_p16_512", "google/siglip2-large-patch16-512"),
        ("siglip2_so400m_p14_224", "google/siglip2-so400m-patch14-224"),
        ("siglip2_so400m_p14_384", "google/siglip2-so400m-patch14-384"),
        ("siglip2_so400m_p16_256", "google/siglip2-so400m-patch16-256"),
        ("siglip2_so400m_p16_384", "google/siglip2-so400m-patch16-384"),
        ("siglip2_so400m_p16_512", "google/siglip2-so400m-patch16-512"),
    ]

    for variant, hf_id in SIGLIP2_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = SiglipModel.from_pretrained(hf_id).eval()
        state = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = SigLIP2ZeroShotClassify.from_weights(variant, load_weights=False)
        transfer_siglip2_weights(keras_model, state)

        out_path = f"{variant}.weights.h5"
        total_gb = keras_model.count_params() * 4 / (1024**3)
        if total_gb > 2.0:
            out_path = f"{variant}.weights.json"
            keras_model.save_weights(out_path, max_shard_size=1.5)
        else:
            keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path} ({total_gb:.2f} GB)")

        del keras_model, hf_model, state
        keras.backend.clear_session()
        gc.collect()
