import json

import keras
import numpy as np
import pytest
from keras import layers, ops

from kerasformers.base import SubclassedBaseModel
from kerasformers.conversion import converted_cache
from kerasformers.conversion.hf_download_utils import (
    LazyStateDict,
    _shard_order_for_plan,
)
from kerasformers.quantization import quantize_model


@keras.saving.register_keras_serializable(package="kerasformers_tests")
class _CacheToy(SubclassedBaseModel):
    def __init__(self, dim=32, depth=2, vocab=64, **kwargs):
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


def test_converted_cache_streams_bounded_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(converted_cache, "SHARD_LIMIT_BYTES", 1024)
    rng = np.random.default_rng(5)
    inputs = {"input_ids": np.array([[2, 4, 8, 16]], dtype="int32")}
    model = _CacheToy()
    model(inputs)
    for weight in model.weights:
        weight.assign(rng.standard_normal(tuple(weight.shape)).astype("float32"))
    quantize_model(model, "int4")
    expected = ops.convert_to_numpy(model(inputs))

    converted_cache.save_converted(model, str(tmp_path), "int4")
    with open(tmp_path / "meta.json") as f:
        meta = json.load(f)
    assert meta["cache_format"] == converted_cache.CACHE_FORMAT_VERSION
    assert len(meta["shards"]) > 1

    restored = converted_cache.load_converted(str(tmp_path), "int4", None)
    actual = ops.convert_to_numpy(restored(inputs))
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="load_dtype"):
        converted_cache.load_converted(str(tmp_path), "int4", "float16")


def test_cache_key_includes_source_revision(monkeypatch):
    class _Model:
        __module__ = "test_models"
        __qualname__ = "Model"

    monkeypatch.setattr(
        converted_cache,
        "_source_identity",
        lambda _: {"kind": "hf", "repo": "org/model", "revision": "a"},
    )
    first = converted_cache.cache_dir(_Model, "hf:org/model", "int4", None, {})
    monkeypatch.setattr(
        converted_cache,
        "_source_identity",
        lambda _: {"kind": "hf", "repo": "org/model", "revision": "b"},
    )
    second = converted_cache.cache_dir(_Model, "hf:org/model", "int4", None, {})
    assert first != second


def test_learned_shard_plan_orders_first_accesses():
    weight_map = {"a": "one", "b": "two", "c": "one", "d": "three"}
    assert _shard_order_for_plan(weight_map, ["b", "a", "d"]) == [
        "two",
        "one",
        "three",
    ]


def test_lazy_state_dict_records_completed_access_plan(tmp_path):
    from safetensors.numpy import save_file

    shard = tmp_path / "one.safetensors"
    save_file({"a": np.array([1]), "b": np.array([2])}, str(shard))
    plan_path = tmp_path / "transfer-plan.json"
    state = LazyStateDict(
        {"a": str(shard), "b": str(shard)},
        shard_of={"a": "one", "b": "one"},
        access_plan_path=str(plan_path),
    )
    np.testing.assert_array_equal(state["b"], np.array([2]))
    np.testing.assert_array_equal(state["a"], np.array([1]))
    state.close()
    assert json.loads(plan_path.read_text())["tensors"] == ["b", "a"]
