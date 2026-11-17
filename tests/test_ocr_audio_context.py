"""Regression tests: whole-word exculpatory matching, OCR text handling and
spoken-transcript scoring.  None of these need torch, Whisper or EasyOCR."""
import sys
import types

import pytest

from model.prototype_scorer import PrototypeHarmScorer
from utils.config import load_policy_config
from utils.text_context import contains_keyword

KEYWORDS = load_policy_config()["exculpatory_keywords"]["keywords"]


# --------------------------------------------------------------------------
# Exculpatory keywords must match whole words only
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "he is contacting you tonight",   # contains "acting"
        "the factory is closed",          # contains "actor"
        "a description of the incident",  # contains "script"
        "property of the state",          # contains "prop"
        "reacting to threats",            # contains "acting"
        "I will kill you",
    ],
)
def test_substrings_do_not_count_as_exculpatory_context(text):
    assert not contains_keyword(text, KEYWORDS)


@pytest.mark.parametrize(
    "text",
    [
        "he is an actor in a short film",
        "Halloween costume party",
        "it's just a prop",
        "two actors rehearsing",
        "auditioning for a role",
        "PHOTO SHOOT",
    ],
)
def test_real_exculpatory_context_still_detected(text):
    assert contains_keyword(text, KEYWORDS)


def test_empty_inputs_are_safe():
    assert not contains_keyword(None, KEYWORDS)
    assert not contains_keyword("", KEYWORDS)
    assert not contains_keyword("anything", [])


# --------------------------------------------------------------------------
# Spoken transcript goes through the same phrase patterns as typed text
# --------------------------------------------------------------------------

def _audio_graph(transcript):
    return {
        "nodes": [
            {"id": "audio_0", "label": "audio", "modality": "audio",
             "text": transcript, "full_label": "audio segment",
             "harmful_signal": False},
        ],
        "edges": [],
    }


def test_spoken_threat_in_transcript_is_flagged():
    r = PrototypeHarmScorer().score(_audio_graph("I will kill you"), {"audio": 0.05})
    assert r["verdict"] == "HARMFUL"
    assert r["modality_scores"]["audio"] >= 0.55
    assert "audio" in r["harmful_modalities"]


def test_benign_transcript_stays_safe():
    r = PrototypeHarmScorer().score(_audio_graph("see you at lunch tomorrow"), {"audio": 0.02})
    assert r["verdict"] == "SAFE"


def test_typed_text_score_is_not_inflated_by_audio_patterns():
    g = _audio_graph("I will kill you")
    r = PrototypeHarmScorer().score(g, {"audio": 0.0})
    assert r["text_pattern_score"] == 0.0


# --------------------------------------------------------------------------
# OCR text flows through the base text node like typed text
# --------------------------------------------------------------------------

def test_ocr_text_on_a_text_node_is_pattern_checked():
    g = {
        "nodes": [
            {"id": "text_0", "label": "text", "modality": "text", "source": "ocr",
             "text": "I will kill you", "full_label": "I will kill you",
             "harmful_signal": False},
            {"id": "image_1", "label": "image", "modality": "image",
             "harmful_signal": False},
        ],
        "edges": [],
    }
    r = PrototypeHarmScorer().score(g, {"text": 0.0, "image": 0.1})
    assert r["text_pattern_score"] >= 0.55
    assert r["verdict"] == "HARMFUL"


# --------------------------------------------------------------------------
# OCREncoder: filtering, ordering and failure reporting (EasyOCR faked)
# --------------------------------------------------------------------------

class _FakeReader:
    def __init__(self, *_a, **_k):
        pass

    def readtext(self, _array, detail=1, paragraph=False):
        box = lambda x, y: [[x, y], [x + 50, y], [x + 50, y + 20], [x, y + 20]]
        return [
            (box(10, 200), "BOTTOM LINE", 0.95),
            (box(10, 10), "top line", 0.90),
            (box(10, 100), "blurry junk", 0.05),   # below ocr_min_confidence
        ]


@pytest.fixture
def fake_easyocr(monkeypatch):
    mod = types.ModuleType("easyocr")
    mod.Reader = _FakeReader
    monkeypatch.setitem(sys.modules, "easyocr", mod)


def test_ocr_orders_lines_and_drops_low_confidence(fake_easyocr):
    PIL = pytest.importorskip("PIL.Image")
    pytest.importorskip("numpy")
    from encoders.ocr_encoder import OCREncoder

    enc = OCREncoder(gpu=False)
    assert enc.available
    out = enc.extract(PIL.new("RGB", (300, 300), "white"))
    assert out["error"] is None
    assert out["text"] == "top line BOTTOM LINE"
    assert [ln["text"] for ln in out["lines"]] == ["top line", "BOTTOM LINE"]


def test_ocr_unavailable_reports_why(monkeypatch):
    monkeypatch.setitem(sys.modules, "easyocr", None)  # import raises
    from encoders.ocr_encoder import OCREncoder

    enc = OCREncoder(gpu=False)
    assert not enc.available
    out = enc.extract(object())
    assert out["available"] is False and out["error"]
    assert out["text"] == ""
