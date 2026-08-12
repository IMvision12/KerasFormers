import os

import pytest
from keras import layers, ops

from kerasformers.base import SubclassedBaseModel
from tests.base.model_test_registry import (
    MODEL_TEST_CONFIGS,
    create_test_input,
    import_model_class,
    instantiate_model,
)

# Substrings marking a sublayer as a vision/audio tower (checked against the
# attribute name and the layer's class name).
_MULTIMODAL_HINTS = (
    "vision",
    "visual",
    "audio",
    "image",
    "aligner",
    "projector",
    "merger",
    "mm_",
    "siglip",
    "speech",
    "high_res",
)


def _multimodal_towers(model):
    """Attribute names of ``model``'s vision/audio tower sublayers."""
    towers = []
    for attr in dir(model):
        try:
            value = getattr(model, attr, None)
        except Exception:
            continue
        if not isinstance(value, layers.Layer):
            continue
        names = (attr.lower(), type(value).__name__.lower())
        if any(hint in name for name in names for hint in _MULTIMODAL_HINTS):
            towers.append(attr)
    return towers


BACKEND = os.environ.get("KERAS_BACKEND", "torch")
MODEL_IDS = list(MODEL_TEST_CONFIGS.keys())

# TF CPU segfaults in tf.matmul for large SAM models (known TF bug).
SKIP_TF_CPU = {"SAMModel", "SAMPromptableSegment", "SAM2PromptableSegment"}


def _skip_if_incompatible(model_name):
    if BACKEND == "tensorflow" and model_name in SKIP_TF_CPU:
        try:
            import tensorflow as tf

            if not tf.config.list_physical_devices("GPU"):
                pytest.skip(f"{model_name}: TF CPU segfaults in matmul")
        except ImportError:
            pass


@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_model_forward_pass(model_name):
    _skip_if_incompatible(model_name)
    config = MODEL_TEST_CONFIGS[model_name]
    model = instantiate_model(config)
    input_data = create_test_input(config)
    output = model(input_data)

    expected = config["expected_output_shape"]
    if expected is None:
        # Models with dynamic output shapes (e.g. SAM): just check it runs
        return

    if isinstance(expected, dict):
        assert isinstance(output, dict), (
            f"{model_name}: expected dict output, got {type(output)}"
        )
        for key, shape in expected.items():
            if shape is not None:
                assert key in output, f"{model_name}: missing output key '{key}'"
                assert output[key].shape == shape, (
                    f"{model_name}[{key}]: expected {shape}, got {output[key].shape}"
                )
    elif isinstance(expected, list):
        assert isinstance(output, (list, tuple)), (
            f"{model_name}: expected list output, got {type(output)}"
        )
        assert len(output) == len(expected), (
            f"{model_name}: expected {len(expected)} outputs, got {len(output)}"
        )
        for i, shape in enumerate(expected):
            if shape is not None:
                assert output[i].shape == shape, (
                    f"{model_name}[{i}]: expected {shape}, got {output[i].shape}"
                )
    else:
        assert output.shape == expected, (
            f"{model_name}: expected {expected}, got {output.shape}"
        )


@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_model_no_nans(model_name):
    _skip_if_incompatible(model_name)
    config = MODEL_TEST_CONFIGS[model_name]
    model = instantiate_model(config)
    input_data = create_test_input(config)
    output = model(input_data)

    if isinstance(output, dict):
        for key, value in output.items():
            has_nans = bool(ops.any(ops.isnan(value)))
            assert not has_nans, f"{model_name}[{key}] contains NaN values"
    elif isinstance(output, (list, tuple)):
        for i, value in enumerate(output):
            has_nans = bool(ops.any(ops.isnan(value)))
            assert not has_nans, f"{model_name}[{i}] contains NaN values"
    else:
        has_nans = bool(ops.any(ops.isnan(output)))
        assert not has_nans, f"{model_name} output contains NaN values"


@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_multimodal_models_override_build_for_transfer(model_name):
    """A subclassed model with a vision/audio tower must override
    ``build_for_transfer``.

    The base ``build_for_transfer`` runs a *text-only* forward, so without an
    override the tower sublayers never materialize. Their weights (e.g. a
    zero-init projector) are then silently skipped when a repo's weights stream
    in via ``from_weights`` / the converted-weight cache, yielding a model that
    is blind to images or audio. This guards a new multimodal port from shipping
    without the override.
    """
    config = MODEL_TEST_CONFIGS[model_name]
    model_cls = import_model_class(config)
    if not issubclass(model_cls, SubclassedBaseModel):
        pytest.skip("build_for_transfer only applies to subclassed models")
    model = instantiate_model(config)
    towers = _multimodal_towers(model)
    if not towers:
        pytest.skip("no multimodal tower in this test config")
    assert (
        type(model).build_for_transfer is not SubclassedBaseModel.build_for_transfer
    ), (
        f"{model_name} has multimodal tower(s) {towers} but does not override "
        f"build_for_transfer(); from_weights / the converted-weight cache would "
        f"build it text-only and leave those tower weights uncreated."
    )
