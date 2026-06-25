import numpy as np
from keras import layers

from .config import resolve_config
from .experts import QuantizedExperts
from .layers import (
    QuantizedDense,
    QuantizedEinsumDense,
    QuantizedEmbedding,
    einsum_contracting_axes,
    get_quantizer,
)
from .quantize import (
    _child_path,
    _is_fused_experts,
    _named_children,
)

# float8 dtypes have no numpy equivalent; they are one byte each.
_DTYPE_BYTES = {"float8_e4m3fn": 1, "float8_e5m2": 1}


def dtype_bytes(dtype):
    """Bytes per element for a keras/numpy/float8 dtype string."""
    dtype = str(dtype)
    if dtype in _DTYPE_BYTES:
        return _DTYPE_BYTES[dtype]
    try:
        return np.dtype(dtype).itemsize
    except TypeError:
        return 4


def tensor_bytes(shape, dtype):
    """Stored size of a tensor of ``shape`` and ``dtype``, in bytes."""
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * dtype_bytes(dtype)


def human_bytes(num_bytes):
    """Format a byte count as a short human string (e.g. ``"4.4 GB"``)."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024


def memory_footprint(model):
    """Actual stored size of every weight in ``model``, in bytes.

    Works on a float model or an already-quantized one (it just sums the real
    dtypes), so it reports true on-disk / in-memory size without any estimation.
    """
    return sum(tensor_bytes(tuple(w.shape), w.dtype) for w in model.weights)


def _quantized_bytes(shape, mode, group_size, axis):
    spec = get_quantizer(mode, group_size).storage_spec(tuple(shape), axis=axis)
    return tensor_bytes(*spec["kernel"]) + tensor_bytes(*spec["scale"])


def _delta(shape, dtype, mode, group_size, axis):
    # Signed byte change from quantizing one weight: quantized (kernel + scale)
    # minus the float it replaces. Negative = saved. Returns 0 if this weight
    # can't be packed for the mode (e.g. int4 needs a single even contracting
    # axis) so the estimate matches what the real swap would leave in float.
    try:
        return _quantized_bytes(shape, mode, group_size, axis) - tensor_bytes(
            shape, dtype
        )
    except (ValueError, ZeroDivisionError):
        return 0


def _layer_delta(layer, config, path):
    delta = 0
    for name, value in _named_children(layer).items():
        if name.startswith("_") or isinstance(
            value,
            (
                QuantizedDense,
                QuantizedEinsumDense,
                QuantizedEmbedding,
                QuantizedExperts,
            ),
        ):
            continue
        child_path = _child_path(path, value, name)
        if isinstance(value, layers.Dense) and value.built:
            mode = config.mode_for(child_path)
            if mode is not None:
                delta += _delta(
                    value.kernel.shape, value.kernel.dtype, mode, config.group_size, 0
                )
        elif isinstance(value, layers.EinsumDense) and value.built:
            mode = config.mode_for(child_path)
            if mode is not None:
                axis = einsum_contracting_axes(value.equation)
                delta += _delta(
                    value.kernel.shape,
                    value.kernel.dtype,
                    mode,
                    config.group_size,
                    axis,
                )
        elif isinstance(value, layers.Embedding) and value.built:
            if config.quantize_embeddings and config.mode_for(child_path) is not None:
                delta += _delta(
                    (value.input_dim, value.output_dim),
                    value.embeddings.dtype,
                    "int8",
                    config.group_size,
                    1,
                )
        elif _is_fused_experts(value):
            mode = config.mode_for(child_path)
            if mode is not None:
                for bank in ("gate_up_proj", "down_proj"):
                    w = getattr(value, bank, None)
                    if w is not None:
                        delta += _delta(w.shape, w.dtype, mode, config.group_size, -1)
        elif isinstance(value, layers.Layer):
            delta += _layer_delta(value, config, child_path)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, layers.Layer):
                    delta += _layer_delta(item, config, _child_path(path, item, name))
    return delta


class MemoryEstimate:
    """Predicted memory footprint of quantizing a model (see :func:`estimate_memory`)."""

    def __init__(self, float_bytes, quantized_bytes, mode, group_size):
        self.float_bytes = float_bytes
        self.quantized_bytes = quantized_bytes
        self.mode = mode
        self.group_size = group_size

    @property
    def saved_bytes(self):
        return self.float_bytes - self.quantized_bytes

    @property
    def compression(self):
        """Float size / quantized size (e.g. ``3.9`` for int8)."""
        return self.float_bytes / self.quantized_bytes if self.quantized_bytes else 0.0

    def fits_in(self, gigabytes):
        """``True`` if the quantized model fits in ``gigabytes`` of memory."""
        return self.quantized_bytes <= gigabytes * (1024**3)

    def summary(self):
        scheme = self.mode + (f"-g{self.group_size}" if self.mode == "int4" else "")
        return (
            f"Quantization memory estimate ({scheme})\n"
            f"  float (current): {human_bytes(self.float_bytes)}\n"
            f"  quantized:       {human_bytes(self.quantized_bytes)}\n"
            f"  saved:           {human_bytes(self.saved_bytes)} "
            f"({self.compression:.2f}x smaller)"
        )

    def __repr__(self):
        return (
            f"MemoryEstimate(mode={self.mode!r}, "
            f"float={human_bytes(self.float_bytes)}, "
            f"quantized={human_bytes(self.quantized_bytes)}, "
            f"compression={self.compression:.2f}x)"
        )


def estimate_memory(model, config="int8", group_size=32):
    """Predict a model's memory footprint **after** quantization, without doing it.

    Walks the same layers :func:`~kerasformers.quantization.quantize_model` would
    convert and sizes their int8 / int4 / fp8 storage from each quantizer's
    ``storage_spec`` (so packed int4 and per-group scales are counted exactly),
    leaving skipped layers (e.g. ``lm_head``) and non-quantizable weights in
    float. Use it to answer "will this fit on my GPU?" before loading.

    Args:
        model: A built model (float weights loaded). The model is **not** mutated.
        config: A :class:`QuantizationConfig`, a bare mode (``"int8"`` / ``"int4"``
            / ``"fp8"``), or a named scheme (``"int4-g128"``).
        group_size: int4 block size when ``config`` is a bare mode string.

    Returns:
        A :class:`MemoryEstimate` with ``float_bytes`` / ``quantized_bytes`` /
        ``compression`` / ``fits_in(gb)`` / ``summary()``.
    """
    config = resolve_config(config, group_size)
    float_bytes = memory_footprint(model)
    quantized_bytes = float_bytes + _layer_delta(model, config, "")
    return MemoryEstimate(float_bytes, quantized_bytes, config.mode, config.group_size)


def quantization_report(model, config=None, group_size=32):
    """Print and return a model's quantization memory profile.

    With ``config`` set, estimates the footprint of quantizing a **float** model
    (the "will it fit?" check). With ``config=None``, reports the **actual**
    footprint of an already-quantized (or float) model.
    """
    if config is not None:
        estimate = estimate_memory(model, config, group_size)
        print(estimate.summary())
        return estimate
    actual = memory_footprint(model)
    print(f"Model footprint: {human_bytes(actual)}")
    return actual
