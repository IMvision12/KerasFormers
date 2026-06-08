from __future__ import annotations

import importlib

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer

# Tokenization is backend-independent: the text -> id mapping is done by the Rust
# ``tokenizers`` library / ``sentencepiece``, and the only Keras touchpoint is the
# final ``ops.convert_to_tensor``, whose integer values are identical on
# torch / tensorflow / jax. So these run on a single backend and compare numpy.

TEXTS = [
    "a quick brown fox jumps over the lazy dog",
    "Hello, World!",
    "tokenization 123 parity",
]
PAIRS = [
    "and a slow green turtle",
    "Goodbye.",
    "the lazy dog sleeps",
]


def _np(x):
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _to_rows(out, pad_id=None):
    """Normalize ANY kerasformers / HF tokenizer output to a list of per-row
    real-token id lists, regardless of the output contract:

    * dict ``input_ids`` / ``token_ids`` (+ ``attention_mask`` / ``padding_mask``),
    * dict with a ragged list-of-lists ``input_ids`` (no padding),
    * a ``(input_ids, attention_mask)`` tuple (sam3).

    Padding is stripped via the mask when present, else via ``pad_id``.
    """
    if isinstance(out, tuple):
        ids, mask = _np(out[0]), _np(out[1])
        if ids.ndim == 1:
            ids, mask = ids[None], mask[None]
        return [[int(t) for t, m in zip(r, mk) if m] for r, mk in zip(ids, mask)]

    ids = out["input_ids"] if "input_ids" in out else out["token_ids"]
    mask = out.get("attention_mask", out.get("padding_mask"))

    if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
        return [[int(t) for t in r] for r in ids]  # ragged, already real tokens

    arr = _np(ids)
    if arr.ndim == 1:
        arr = arr[None]
    rows = [[int(t) for t in r] for r in arr]
    if mask is not None:
        m = _np(mask)
        if m.ndim == 1:
            m = m[None]
        rows = [[t for t, mm in zip(r, row_m) if mm] for r, row_m in zip(rows, m)]
    elif pad_id is not None:
        rows = [[t for t in r if t != pad_id] for r in rows]
    return rows


def _hf_rows(hf, add_special):
    return [
        [int(x) for x in hf(t, add_special_tokens=add_special)["input_ids"]]
        for t in TEXTS
    ]


def _assert_rows(name, ours_rows, hf_rows):
    assert len(ours_rows) == len(hf_rows), f"{name}: row count mismatch"
    for i, (o, h) in enumerate(zip(ours_rows, hf_rows)):
        assert o == h, (
            f"{name}: text[{i}] ids differ from HF\n  ours ({len(o)}): {o}\n"
            f"  hf   ({len(h)}): {h}"
        )


def _build(module, cls_name, repo):
    """Construct the kerasformers tokenizer directly (validates the shipped
    release vocab); fall back to ``from_hf(repo)`` if the release isn't
    available so the parity check still runs."""
    cls = getattr(importlib.import_module(module), cls_name)
    try:
        return cls()
    except Exception:
        if repo and hasattr(cls, "from_hf"):
            return cls.from_hf(repo)
        raise


# name -> (submodule, class, hf_repo | None=use ours.hf_id, add_special, pad_attr)
SPECS = {
    "bert": (
        "kerasformers.models.bert.bert_tokenizer",
        "BertTokenizer",
        "bert-base-uncased",
        True,
        None,
    ),
    "clip": (
        "kerasformers.models.clip.clip_tokenizer",
        "CLIPTokenizer",
        "openai/clip-vit-base-patch16",
        True,
        None,
    ),
    "deberta": (
        "kerasformers.models.deberta.deberta_tokenizer",
        "DebertaTokenizer",
        "microsoft/deberta-base",
        True,
        None,
    ),
    "deberta_v2": (
        "kerasformers.models.deberta_v2.deberta_v2_tokenizer",
        "DebertaV2Tokenizer",
        "microsoft/deberta-v2-xlarge",
        True,
        None,
    ),
    "deberta_v3": (
        "kerasformers.models.deberta_v3.deberta_v3_tokenizer",
        "DebertaV3Tokenizer",
        "microsoft/deberta-v3-base",
        True,
        None,
    ),
    "gpt": (
        "kerasformers.models.gpt.gpt_tokenizer",
        "GptTokenizer",
        "openai-community/openai-gpt",
        False,
        None,
    ),
    "gpt2": (
        "kerasformers.models.gpt2.gpt2_tokenizer",
        "GPT2Tokenizer",
        "openai-community/gpt2",
        False,
        None,
    ),
    "gpt_oss": (
        "kerasformers.models.gpt_oss.gpt_oss_tokenizer",
        "GptOssTokenizer",
        None,
        False,
        None,
    ),
    "granite_speech": (
        "kerasformers.models.granite_speech.granite_speech_tokenizer",
        "GraniteSpeechTokenizer",
        "ibm-granite/granite-speech-3.3-2b",
        False,
        None,
    ),
    "granite_speech_plus": (
        "kerasformers.models.granite_speech_plus.granite_speech_plus_tokenizer",
        "GraniteSpeechPlusTokenizer",
        "ibm-granite/granite-speech-4.1-2b-plus",
        False,
        None,
    ),
    "metaclip2": (
        "kerasformers.models.metaclip2.metaclip2_tokenizer",
        "MetaClip2Tokenizer",
        "facebook/metaclip-2-worldwide-huge-378",
        True,
        None,
    ),
    "metaclip2_mt5": (
        "kerasformers.models.metaclip2.metaclip2_mt5_tokenizer",
        "MetaClip2Mt5Tokenizer",
        "google/mt5-base",
        True,
        None,
    ),
    "moonshine": (
        "kerasformers.models.moonshine.moonshine_tokenizer",
        "MoonshineTokenizer",
        "UsefulSensors/moonshine-tiny",
        False,
        None,
    ),
    "qwen2": (
        "kerasformers.models.qwen2.qwen2_tokenizer",
        "Qwen2Tokenizer",
        None,
        False,
        None,
    ),
    "qwen2_vl": (
        "kerasformers.models.qwen2_vl.qwen2_vl_tokenizer",
        "Qwen2VLTokenizer",
        None,
        False,
        None,
    ),
    "qwen3": (
        "kerasformers.models.qwen3.qwen3_tokenizer",
        "Qwen3Tokenizer",
        None,
        False,
        None,
    ),
    "qwen3_5": (
        "kerasformers.models.qwen3_5.qwen3_5_tokenizer",
        "Qwen3_5Tokenizer",
        None,
        False,
        None,
    ),
    "roberta": (
        "kerasformers.models.roberta.roberta_tokenizer",
        "RobertaTokenizer",
        "roberta-base",
        True,
        None,
    ),
    "siglip": (
        "kerasformers.models.siglip.siglip_tokenizer",
        "SigLIPTokenizer",
        "google/siglip-base-patch16-224",
        True,
        "pad_token_id",
    ),
    "siglip2": (
        "kerasformers.models.siglip2.siglip2_tokenizer",
        "SigLIP2Tokenizer",
        "google/siglip2-base-patch16-224",
        True,
        "pad_token_id",
    ),
    "speech2text": (
        "kerasformers.models.speech2text.speech2text_tokenizer",
        "Speech2TextTokenizer",
        "facebook/s2t-small-librispeech-asr",
        True,
        "pad_token_id",
    ),
    "whisper": (
        "kerasformers.models.whisper.whisper_tokenizer",
        "WhisperTokenizer",
        "openai/whisper-tiny",
        False,
        None,
    ),
    "xlm_roberta": (
        "kerasformers.models.xlm_roberta.xlm_roberta_tokenizer",
        "XLMRobertaTokenizer",
        "xlm-roberta-base",
        True,
        None,
    ),
}


@pytest.mark.parametrize("name", list(SPECS.keys()))
def test_tokenizer_hf_parity(name):
    if name == "metaclip2_mt5":
        pytest.skip("MetaCLIP2 mT5 text tokenizer: HF source repo not pinned")
    module, cls_name, repo, add_special, pad_attr = SPECS[name]
    try:
        ours_tok = _build(module, cls_name, repo)
    except Exception as e:  # release not uploaded and no HF fallback
        pytest.skip(f"cannot construct {name}: {type(e).__name__}: {e}")

    hf_repo = repo or getattr(ours_tok, "hf_id", None)
    if hf_repo is None:
        pytest.skip(f"{name}: no HF repo to compare against")
    try:
        hf = AutoTokenizer.from_pretrained(hf_repo)
    except Exception as e:
        pytest.skip(f"{name}: HF tokenizer for {hf_repo!r} unavailable: {e}")

    if name == "siglip":
        # SigLIP's pad id == eos id and there's no attention mask, so stripping
        # pads by value would drop the real trailing eos. Compare full padded
        # arrays instead (HF padded to the same length).
        ours_ids = _np(ours_tok(TEXTS)["input_ids"])
        hf_ids = _np(
            hf(
                TEXTS,
                padding="max_length",
                max_length=ours_ids.shape[1],
                add_special_tokens=add_special,
                return_tensors="np",
            )["input_ids"]
        )
        assert ours_ids.shape == hf_ids.shape and np.array_equal(ours_ids, hf_ids), (
            "siglip: padded input_ids differ from HF"
        )
        return

    pad_id = getattr(ours_tok, pad_attr) if pad_attr else None
    ours_rows = _to_rows(ours_tok(TEXTS), pad_id)
    hf_rows = _hf_rows(hf, add_special)
    _assert_rows(name, ours_rows, hf_rows)


def test_sam3_clip_tokenizer_vs_clip():
    """SAM3's CLIP text tokenizer returns a ``(input_ids, attention_mask)`` tuple
    via ``encode`` (no HF ``AutoTokenizer`` of its own); compare its real tokens
    against the OpenAI CLIP tokenizer it mirrors."""
    from kerasformers.models.sam3.sam3_clip_tokenizer import SAM3CLIPTokenizer

    try:
        ours = SAM3CLIPTokenizer()
    except Exception as e:
        pytest.skip(f"cannot construct sam3_clip: {e}")
    hf = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    ours_rows = _to_rows(ours.encode(TEXTS))
    hf_rows = _hf_rows(hf, add_special=True)
    _assert_rows("sam3_clip", ours_rows, hf_rows)


def test_bert_token_type_ids_pairs():
    """Exercise ``token_type_ids`` (0 for segment A, 1 for segment B) on a
    BERT text-pair batch, vs HF."""
    from kerasformers.models.bert.bert_tokenizer import BertTokenizer

    ours = BertTokenizer()(TEXTS, text_pair=PAIRS)
    hf = AutoTokenizer.from_pretrained("bert-base-uncased")(
        TEXTS, PAIRS, padding=True, return_tensors="np"
    )
    o_mask = _np(ours["attention_mask"]).astype(bool)
    o_types, h_types = _np(ours["token_type_ids"]), _np(hf["token_type_ids"])
    assert np.array_equal(o_types[o_mask], h_types[o_mask]), (
        "bert pairs: token_type_ids differ"
    )
    assert int(o_types[o_mask].max()) == 1, "expected a second segment (type id 1)"
