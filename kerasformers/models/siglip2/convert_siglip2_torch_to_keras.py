import gc
from typing import Dict

import keras
import numpy as np
from transformers import SiglipModel

from kerasformers.models.siglip.convert_siglip_torch_to_keras import (
    transfer_siglip_weights,
)
from kerasformers.models.siglip2 import SigLIP2ZeroShotClassify


def transfer_siglip2_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    transfer_siglip_weights(keras_model, hf_state_dict)


if __name__ == "__main__":
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

        total_params = sum(int(np.prod(w.shape)) for w in keras_model.weights)
        total_gb = (total_params * 4) / (1024**3)
        if total_gb > 2:
            out_path = f"{variant}.weights.json"
            keras_model.save_weights(out_path, max_shard_size=2)
            print(f"  Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
        else:
            out_path = f"{variant}.weights.h5"
            keras_model.save_weights(out_path)
            print(f"  Saved -> {out_path} (~{total_gb:.2f} GB)")

        del keras_model, hf_model, state
        keras.backend.clear_session()
        gc.collect()
