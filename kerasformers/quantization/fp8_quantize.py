import keras
import ml_dtypes
from keras import ops

from kerasformers.base import Quantizer, normalize_axes

FP8_DTYPE = "float8_e4m3fn"
FP8_MAX = float(ml_dtypes.finfo(FP8_DTYPE).max)  # 448.0


def fp8_supported():
    """fp8 (e4m3) storage works on the torch and jax backends only; tensorflow
    lacks the float8 casts needed to store/quantize the weights."""
    return keras.backend.backend() in ("torch", "jax")


@keras.saving.register_keras_serializable(package="kerasformers")
class Fp8Quantizer(Quantizer):
    """Per-channel float8 (e4m3) weight-only quantizer (~4x smaller).

    Stores the weight in the native ``float8_e4m3fn`` dtype (1 byte) with one
    float32 scale per output channel (``scale = max|w| / 448`` over the
    contracting ``axis``). At the same footprint as int8 the floating-point grid
    (4 exponent / 3 mantissa bits) often tracks weight distributions with wide
    dynamic range better than uniform int8. **torch / jax only** — tensorflow
    lacks the float8 casts. ``axis`` may be an int or a tuple.
    """

    mode = "fp8"

    def quantize(self, weight, axis=0):
        """Quantize ``weight`` along ``axis`` -> ``(fp8 weight, float32 scale)``.

        The per-channel scale maps each channel into the e4m3 range (|x| <= 448)
        before casting; ``scale`` keeps reduced axes as size 1 for broadcasting.
        """
        weight = ops.convert_to_tensor(weight)
        amax = ops.max(ops.abs(weight), axis=axis, keepdims=True)
        scale = ops.maximum(amax / FP8_MAX, keras.config.epsilon())
        q = ops.clip(weight / scale, -FP8_MAX, FP8_MAX)
        return ops.cast(q, FP8_DTYPE), ops.cast(scale, "float32")

    def dequantize(self, packed, scale, axis=0, dtype=None):
        """Reconstruct the float weight, in ``dtype`` (defaults to ``scale``'s)."""
        dtype = dtype or scale.dtype
        return ops.cast(packed, dtype) * ops.cast(scale, dtype)

    def storage_spec(self, weight_shape, axis=0):
        axes = normalize_axes(axis, len(weight_shape))
        scale_shape = tuple(1 if i in axes else d for i, d in enumerate(weight_shape))
        return {
            "kernel": (tuple(weight_shape), FP8_DTYPE),
            "scale": (scale_shape, "float32"),
        }
