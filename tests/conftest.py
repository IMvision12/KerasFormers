import gc
import os

import pytest


@pytest.fixture(autouse=True)
def _release_backend_state():
    """Release per-test Keras / JAX state to keep CI memory bounded.

    Each parametrized model in the integration suite triggers fresh
    XLA / TF function tracing. Without an explicit teardown the JIT
    cache, compiled HLO modules, and dead Keras layers accumulate
    across the 300+ tests and the JAX matrix entry hits the
    ubuntu-latest runner's 7 GB RAM / 60 min timeout (visible as
    process SIGTERM, exit code 143).
    """
    yield
    try:
        import keras

        keras.utils.clear_session()
        if keras.config.backend() == "jax":
            import jax

            jax.clear_caches()
    except Exception:
        pass
    gc.collect()


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="Keras backend to use: torch, tensorflow, jax, numpy",
    )
    parser.addoption(
        "--data-format",
        action="store",
        default=None,
        help="Image data format: channels_first, channels_last",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "serialization: serialization roundtrip tests")
    config.addinivalue_line("markers", "saving: model save/load tests")
    config.addinivalue_line("markers", "data_format: channels first/last tests")
    config.addinivalue_line(
        "markers",
        "link_validation: weight URL + download tests (requires network)",
    )
    config.addinivalue_line("markers", "slow: slow tests excluded from default runs")
    config.addinivalue_line("markers", "gpu: tests that require GPU (skipped on CI)")


def is_gpu_available():
    import keras

    backend = keras.config.backend()
    if backend == "tensorflow":
        try:
            import tensorflow as tf

            return len(tf.config.list_physical_devices("GPU")) > 0
        except ImportError:
            return False
    if backend == "torch":
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False
    return False


def skip_if_no_gpu(reason="This test requires GPU"):
    return pytest.mark.skipif(not is_gpu_available(), reason=reason)


def skip_tf_channels_first():
    import keras

    if keras.config.backend() == "tensorflow" and not is_gpu_available():
        pytest.skip("TF channels_first conv2d requires GPU (cuDNN)")


def skip_numpy_backend():
    import keras

    if keras.config.backend() == "numpy":
        pytest.skip("numpy backend doesn't support this operation")


@pytest.fixture
def backend():
    """Return the current Keras backend name."""
    return os.environ.get("KERAS_BACKEND", "torch")
