"""Xception weight transfer.

The kmodels :class:`Xception` model implements the *original* Keras
(Chollet 2017) Xception architecture, not timm's *aligned* Xception
family (``xception41``/``65``/``71``). timm does not host weights for
the original architecture, so this module exposes a
``transfer_xception_weights`` shim that raises ``NotImplementedError``
to make the mismatch explicit, and a ``__main__`` block that
reproduces the legacy ``keras.applications.Xception`` -> kmodels
conversion that produced the release file in ``XCEPTION_WEIGHTS``.
"""

from typing import Dict

import numpy as np


def transfer_xception_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    """Transfer a timm Xception state-dict into a kmodels Xception (unsupported).

    Raises:
        NotImplementedError: kmodels' Xception is the original Keras
            architecture; timm's Xception family is the Aligned Xception
            variant with a different block layout. No 1:1 weight mapping
            exists. Use the kmodels release weight via
            :meth:`Xception.from_weights("xception_in1k")` instead.
    """
    raise NotImplementedError(
        "transfer_xception_weights: kmodels.Xception is the original-Keras "
        "(Chollet 2017) Xception. timm's xception41/65/71/p families are "
        "Aligned Xception variants with a different block layout. There is "
        "no 1:1 weight mapping. Use Xception.from_weights('xception_in1k') "
        "for the converted keras.applications checkpoint."
    )


if __name__ == "__main__":
    # Re-create the kmodels release ``keras_org_xception.weights.h5`` from
    # ``keras.applications.Xception``. Run once to regenerate the release.
    import keras

    from kmodels.models.xception import XceptionClassify

    original_model = keras.applications.Xception(
        input_shape=(299, 299, 3),
        classifier_activation="linear",
        weights="imagenet",
        include_top=True,
    )

    custom_model = XceptionClassify.from_weights("xception_in1k", load_weights=False)
    custom_model.set_weights(original_model.get_weights())
    custom_model.save_weights("xception_in1k.weights.h5")
    print("Saved -> xception_in1k.weights.h5")
