from __future__ import annotations

import keras
import numpy as np
import pytest
from PIL import Image

transformers = pytest.importorskip("transformers")

from transformers import AutoProcessor

MM_TEXTS = ["a photo of a cat", "two dogs running on the beach"]


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "numpy"):
        return x.numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return keras.ops.convert_to_numpy(x)


def _to_channels_last(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        return np.transpose(arr, (0, 2, 3, 1))
    return arr


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_channels_last(a)
    b = _to_channels_last(b)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _rgb(side, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((side, side, 3)) * 255).astype("uint8")


def _strip_pad(ids, mask):
    ids = np.asarray(_as_numpy(ids))
    mask = np.asarray(_as_numpy(mask)).astype(bool)
    return [[int(t) for t, m in zip(row, mrow) if m] for row, mrow in zip(ids, mask)]


def _auto_processor(repo):
    try:
        return AutoProcessor.from_pretrained(repo)
    except Exception as e:
        pytest.skip(f"HF AutoProcessor for {repo!r} unavailable: {e}")


def _legs(cls, repo):
    return [("native", cls()), ("from_hf", cls.from_weights(f"hf:{repo}"))]


def test_clip_processor_three_way():
    from kerasformers.models.clip.clip_processor import CLIPProcessor

    repo = "openai/clip-vit-base-patch16"
    hf = _auto_processor(repo)
    img = _rgb(224)
    h = hf(
        text=MM_TEXTS, images=Image.fromarray(img), padding=True, return_tensors="np"
    )
    hf_rows = _strip_pad(h["input_ids"], h["attention_mask"])
    for leg, ours in _legs(CLIPProcessor, repo):
        o = ours(text=MM_TEXTS, images=img)
        assert _strip_pad(o["input_ids"], o["attention_mask"]) == hf_rows, (
            f"clip[{leg}]: input_ids differ from HF"
        )
        diff = _max_diff(_as_numpy(o["images"]), h["pixel_values"])
        assert diff < 1e-4, f"clip[{leg}]: pixel max|diff|={diff:.3e}"
        print(f"[{leg:>7} clip processor      ] ids ok, pixel max|diff|={diff:.3e}")


def test_siglip_processor_three_way():
    from kerasformers.models.siglip.siglip_processor import SigLIPProcessor

    repo = "google/siglip-base-patch16-224"
    hf = _auto_processor(repo)
    img = _rgb(224)
    h = hf(
        text=MM_TEXTS,
        images=Image.fromarray(img),
        padding="max_length",
        max_length=64,
        return_tensors="np",
    )
    for leg, ours in _legs(SigLIPProcessor, repo):
        o = ours(text=MM_TEXTS, images=img)
        # SigLIP pads with the eos id and returns no attention mask, so compare
        # the full fixed-length id arrays.
        assert np.array_equal(np.asarray(_as_numpy(o["input_ids"])), h["input_ids"]), (
            f"siglip[{leg}]: input_ids differ from HF"
        )
        diff = _max_diff(_as_numpy(o["images"]), h["pixel_values"])
        assert diff < 1e-4, f"siglip[{leg}]: pixel max|diff|={diff:.3e}"
        print(f"[{leg:>7} siglip processor    ] ids ok, pixel max|diff|={diff:.3e}")


def test_owlvit_processor_three_way():
    from kerasformers.models.owlvit.owlvit_processor import OwlViTProcessor

    repo = "google/owlvit-base-patch32"
    hf = _auto_processor(repo)
    img = _rgb(768)
    queries = [["a photo of a cat", "a photo of a dog"]]
    h = hf(text=queries, images=Image.fromarray(img), return_tensors="np")
    for leg, ours in _legs(OwlViTProcessor, repo):
        o = ours(text=queries, images=img)
        # Both pad to the fixed query length (16) with the "!" pad id.
        assert np.array_equal(np.asarray(_as_numpy(o["input_ids"])), h["input_ids"]), (
            f"owlvit[{leg}]: input_ids differ from HF"
        )
        diff = _max_diff(_as_numpy(o["pixel_values"]), h["pixel_values"])
        assert diff < 1e-4, f"owlvit[{leg}]: pixel max|diff|={diff:.3e}"
        print(f"[{leg:>7} owlvit processor    ] ids ok, pixel max|diff|={diff:.3e}")


def test_whisper_processor_three_way():
    from kerasformers.models.whisper.whisper_processor import WhisperProcessor

    repo = "openai/whisper-tiny"
    hf = _auto_processor(repo)
    t = np.arange(16000 * 2, dtype="float32") / 16000.0
    wave = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype("float32")
    h_feat = hf.feature_extractor(wave, sampling_rate=16000, return_tensors="np")[
        "input_features"
    ]
    h_ids = [hf.tokenizer(x, add_special_tokens=False)["input_ids"] for x in MM_TEXTS]
    for leg, ours in _legs(WhisperProcessor, repo):
        o = ours(audio=wave, text=MM_TEXTS)
        o_feat = np.asarray(_as_numpy(o["input_features"]))
        assert o_feat.shape == h_feat.shape, (
            f"whisper[{leg}]: features shape {o_feat.shape} vs HF {h_feat.shape}"
        )
        diff = float(np.max(np.abs(o_feat - h_feat)))
        assert diff < 5e-3, f"whisper[{leg}]: features max|diff|={diff:.3e}"
        assert _strip_pad(o["input_ids"], o["attention_mask"]) == h_ids, (
            f"whisper[{leg}]: input_ids differ from HF"
        )
        print(f"[{leg:>7} whisper processor   ] ids ok, mel max|diff|={diff:.3e}")
