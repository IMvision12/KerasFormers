"""Compare legacy and streaming on-the-fly quantized loading on a laptop.

Run with ``KERAS_BACKEND=torch python benchmarks/benchmark_quantized_loading.py``.
It uses a local 18.9M-parameter model, avoids network time, and prints JSON.
"""

import gc
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras
import numpy as np
import psutil
from keras import layers, ops
from safetensors.numpy import save_file

from kerasformers.base import SubclassedBaseModel
from kerasformers.conversion import converted_cache
from kerasformers.quantization import quantize_and_load


@keras.saving.register_keras_serializable(package="kerasformers_benchmarks")
class TinyCacheModel(SubclassedBaseModel):
    def __init__(self, dim=2048, depth=4, vocab=512, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.vocab = vocab
        self.embedding = layers.Embedding(vocab, dim, name="embedding")
        for index in range(depth):
            setattr(self, f"block_{index}", layers.Dense(dim, name=f"block_{index}"))
        self.output = layers.Dense(vocab, name="output")

    def call(self, inputs):
        x = self.embedding(inputs["input_ids"])
        for index in range(self.depth):
            x = getattr(self, f"block_{index}")(x)
        return self.output(x)

    def get_config(self):
        return {
            "dim": self.dim,
            "depth": self.depth,
            "vocab": self.vocab,
            **super().get_config(),
        }


def measure(fn):
    process = psutil.Process()
    before = process.memory_info().rss
    peak = before
    running = True

    def sample():
        nonlocal peak
        while running:
            peak = max(peak, process.memory_info().rss)
            time.sleep(0.002)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        result = fn()
    finally:
        elapsed = time.perf_counter() - start
        running = False
        sampler.join()
    return result, {
        "seconds": round(elapsed, 3),
        "peak_extra_mib": round((peak - before) / 2**20, 1),
    }


def legacy_cache_write(model, directory, shard_limit):
    """The pre-streaming cache writer, retained only for this benchmark."""
    weights = list(model.weights)
    arrays = [np.ascontiguousarray(ops.convert_to_numpy(weight)) for weight in weights]
    current, current_bytes, shard_index = {}, 0, 0
    for index, array in enumerate(arrays):
        if current and current_bytes + array.nbytes > shard_limit:
            save_file(
                current,
                os.path.join(directory, f"weights-{shard_index:05d}.safetensors"),
            )
            current, current_bytes, shard_index = {}, 0, shard_index + 1
        current[f"{index:06d}"] = array
        current_bytes += array.nbytes
    save_file(
        current, os.path.join(directory, f"weights-{shard_index:05d}.safetensors")
    )


def main():
    rng = np.random.default_rng(42)
    inputs = {"input_ids": np.ones((1, 8), dtype="int32")}
    reference = TinyCacheModel()
    reference(inputs)
    source = {
        weight.path.split("/", 1)[1]: rng.standard_normal(tuple(weight.shape)).astype(
            "float32"
        )
        for weight in reference.weights
    }

    def transfer(model, state):
        if not model.built:
            model(inputs)
        for weight in model.weights:
            weight.assign(state[weight.path.split("/", 1)[1]])

    model = TinyCacheModel()
    _, no_float_load = measure(
        lambda: quantize_and_load(model, "int4", transfer, source)
    )
    params = sum(int(np.prod(weight.shape)) for weight in reference.weights)
    shard_limit = 8 * 1024**2
    before_dir = tempfile.mkdtemp(prefix="kf_legacy_cache_")
    after_dir = tempfile.mkdtemp(prefix="kf_stream_cache_")
    try:
        _, legacy_cache = measure(
            lambda: legacy_cache_write(model, before_dir, shard_limit)
        )
        old_limit = converted_cache.SHARD_LIMIT_BYTES
        converted_cache.SHARD_LIMIT_BYTES = shard_limit
        try:
            _, streaming_cache = measure(
                lambda: converted_cache.save_converted(model, after_dir, "int4")
            )
        finally:
            converted_cache.SHARD_LIMIT_BYTES = old_limit
        restored, cache_reload = measure(
            lambda: converted_cache.load_converted(after_dir, "int4", None)
        )
        output = ops.convert_to_numpy(restored(inputs))
        cache_mib = round(
            sum(
                os.path.getsize(os.path.join(after_dir, name))
                for name in os.listdir(after_dir)
            )
            / 2**20,
            2,
        )
    finally:
        shutil.rmtree(before_dir)
        shutil.rmtree(after_dir)

    print(
        json.dumps(
            {
                "parameters": params,
                "no_float_int4_load": no_float_load,
                "cache_write_before": legacy_cache,
                "cache_write_after": streaming_cache,
                "cache_reload_after": cache_reload,
                "cache_mib": cache_mib,
                "output_shape": list(output.shape),
                "output_finite": bool(np.isfinite(output).all()),
                "benchmark_cache_shard_mib": shard_limit // 2**20,
            },
            sort_keys=True,
        )
    )
    gc.collect()


if __name__ == "__main__":
    main()
