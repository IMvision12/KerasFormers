import copy

from keras import layers

from .config import resolve_config
from .fp8_quantize import fp8_supported
from .quantize import _apply_quantization, _named_children, dequantize_model

# Output-head name fragments kept in full precision by default — the
# `get_keys_to_not_convert` idea: output projections are accuracy-sensitive.
HEAD_PATTERNS = (
    "lm_head",
    "classifier",
    "logits",
    "score",
    "prediction",
    "qa_outputs",
    "output_head",
    "seq_relationship",
)


def detect_modules_to_not_convert(model):
    """Auto-detect output-head ``Dense`` layers to keep in full precision.

    The kerasformers analogue of ``transformers.get_keys_to_not_convert`` — scans
    the model (subclassed attributes and functional ``.layers``) for Dense layers
    whose name looks like an output head.
    """
    found, seen, stack = [], set(), [model]
    while stack:
        layer = stack.pop()
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        if isinstance(layer, layers.Dense) and any(
            p in layer.name.lower() for p in HEAD_PATTERNS
        ):
            found.append(layer.name)
        for value in _named_children(layer).values():
            if isinstance(value, layers.Layer):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(v for v in value if isinstance(v, layers.Layer))
        for value in getattr(layer, "layers", None) or []:
            stack.append(value)
    return found


def _augmented(config, extra_skips):
    new = copy.copy(config)
    new.skip_modules = tuple(
        dict.fromkeys(list(config.skip_modules) + list(extra_skips))
    )
    return new


# Per-model-class module patches applied *before* quantization — the kerasformers
# analogue of transformers' MODULES_TO_PATCH_FOR_QUANTIZATION. Maps a layer class
# name -> {"replace": (layer)->layer, "methods": (...)}. Empty by default (our
# experts auto-quantize via duck-typing); this is the extension hook for a custom
# layer that needs a quantization-friendly variant swapped in first.
MODULE_PATCH_REGISTRY = {}


def register_quantization_patch(
    class_name, replace_fn, methods=("int8", "int4", "fp8")
):
    """Register a pre-quantization swap for ``class_name`` (see MODULE_PATCH_REGISTRY)."""
    MODULE_PATCH_REGISTRY[class_name] = {
        "replace": replace_fn,
        "methods": tuple(methods),
    }


def apply_module_patches(model, quant_method):
    """Swap registered layer classes for their quantization-friendly variants."""
    if not MODULE_PATCH_REGISTRY:
        return
    from .quantize import _swap

    def walk(parent):
        for name, value in list(_named_children(parent).items()):
            if name.startswith("_") or not isinstance(value, layers.Layer):
                continue
            patch = MODULE_PATCH_REGISTRY.get(type(value).__name__)
            if patch and quant_method in patch["methods"]:
                _swap(parent, name, value, patch["replace"](value))
            else:
                walk(value)

    walk(model)


class BaseQuantizer:
    """Lifecycle orchestrator for a quantization method (transformers ``HfQuantizer``).

    A method is the pipeline ``validate_environment`` ->
    ``get_modules_to_not_convert`` (config skips + auto-detected output heads) ->
    ``preprocess_model`` (swap layers in place / clone) -> ``postprocess_model``
    (record the config on the model). Subclasses set ``quant_method`` and may
    override ``validate_environment`` (e.g. fp8's backend check).
    """

    requires_calibration = False
    quant_method = None

    def __init__(self, config):
        self.config = config
        self.modules_to_not_convert = []

    def validate_environment(self):
        """Raise if the method can't run on the current backend / config."""
        return

    def update_dtype(self, dtype):
        return dtype

    def get_modules_to_not_convert(self, model):
        return list(
            dict.fromkeys(
                list(self.config.skip_modules) + detect_modules_to_not_convert(model)
            )
        )

    def preprocess_model(self, model):
        apply_module_patches(model, self.quant_method)
        return _apply_quantization(model, self.config)

    def postprocess_model(self, model):
        model._quantization_config = self.config
        return model

    def quantize(self, model):
        """Run the full lifecycle and return the quantized model."""
        self.validate_environment()
        self.modules_to_not_convert = self.get_modules_to_not_convert(model)
        extra = [
            s for s in self.modules_to_not_convert if s not in self.config.skip_modules
        ]
        if extra:
            self.config = _augmented(self.config, extra)
        model = self.preprocess_model(model)
        return self.postprocess_model(model)

    def dequantize(self, model):
        return dequantize_model(model)

    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return False

    @property
    def is_compileable(self):
        return True

    def __repr__(self):
        return f"{type(self).__name__}({self.config!r})"


class Int8Quantization(BaseQuantizer):
    quant_method = "int8"


class Int4Quantization(BaseQuantizer):
    quant_method = "int4"


class Fp8Quantization(BaseQuantizer):
    quant_method = "fp8"

    def validate_environment(self):
        if not fp8_supported():
            raise ValueError(
                "fp8 quantization requires the torch or jax backend "
                "(tensorflow lacks float8 casts)."
            )


_QUANTIZER_FOR_METHOD = {
    "int8": Int8Quantization,
    "int4": Int4Quantization,
    "fp8": Fp8Quantization,
}


class AutoQuantizer:
    """Dispatch a config to its method orchestrator (transformers ``AutoHfQuantizer``)."""

    @staticmethod
    def from_config(config, group_size=32):
        config = resolve_config(config, group_size)
        return _QUANTIZER_FOR_METHOD[config.quant_method](config)
