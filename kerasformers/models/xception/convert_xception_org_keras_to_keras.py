from typing import Dict

import keras
import numpy as np

from kerasformers.models.xception import XceptionImageClassify


def transfer_xception_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    raise NotImplementedError(
        "transfer_xception_weights: kerasformers.XceptionImageClassify is the "
        "original-Keras (Chollet 2017) Xception. timm's xception41/65/71/p "
        "families are Aligned Xception variants with a different block layout. "
        "There is no 1:1 weight mapping. Use "
        "XceptionImageClassify.from_weights('xception_in1k') for the converted "
        "keras.applications checkpoint."
    )


if __name__ == "__main__":
    original_model = keras.applications.Xception(
        input_shape=(299, 299, 3),
        classifier_activation="linear",
        weights="imagenet",
        include_top=True,
    )

    custom_model = XceptionImageClassify.from_weights(
        "xception_in1k", load_weights=False, include_normalization=False
    )
    custom_model.set_weights(original_model.get_weights())
    custom_model.save_weights("xception_in1k.weights.h5")
    print("Saved -> xception_in1k.weights.h5")
