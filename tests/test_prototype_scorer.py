"""PrototypeHarmScorer: the API's scorer must see the sentence and stay consistent."""
from model.prototype_scorer import PrototypeHarmScorer


def _graph(sentence=None, harmful=(), full_label=None):
    """Nodes shaped like the text encoder's output after the API tags them."""
    base = {"id": "text_0", "label": "text", "modality": "text",
            "full_label": full_label or sentence or "", "harmful_signal": False}
    if sentence is not None:
        base["text"] = sentence
    nodes = [base]
    for i, (cat, title, conf) in enumerate(harmful, start=1):
        nodes.append({"id": f"text_{i}", "label": cat, "full_label": title,
                      "modality": "text", "harmful_signal": True,
                      "policy_category": cat, "confidence": conf})
    return {"nodes": nodes, "edges": []}


def test_explicit_threat_is_flagged():
    g = _graph("I will kill you", [("violence", "Violence", 0.9)])
    r = PrototypeHarmScorer().score(g, {"text": 0.91})
    assert r["verdict"] == "HARMFUL" and r["harm_score"] >= 0.55
    assert "violence" in " ".join(r["text_pattern_categories"]).lower() or r["text_pattern_score"] > 0


def test_pattern_found_even_from_encoder_preview_only():
    g = _graph(None, full_label="i'm going to kill you")
    assert PrototypeHarmScorer().score(g, {"text": 0.2})["text_pattern_score"] >= 0.55


def test_strong_toxicity_without_phrase_match_is_flagged():
    g = _graph("you absolute waste of space", [("harassment", "Bullying", 0.92)])
    r = PrototypeHarmScorer().score(g, {"text": 0.92})
    assert r["verdict"] == "HARMFUL"


def test_moderate_toxicity_and_benign_text_stay_safe():
    s = PrototypeHarmScorer()
    assert s.score(_graph("that was rude"), {"text": 0.5})["verdict"] == "SAFE"
    assert s.score(_graph("nice weather today"), {"text": 0.02})["verdict"] == "SAFE"


def test_explanation_agrees_with_verdict_and_cites_evidence():
    g = _graph("I will kill you", [("violence", "Violence", 0.9)])
    r = PrototypeHarmScorer().score(g, {"text": 0.91})
    assert r["explanation"].startswith(r["verdict"] + ":")
    assert "Violence" in r["explanation"] and "text 91" in r["explanation"]
    safe = PrototypeHarmScorer().score(_graph("hello"), {"text": 0.01})
    assert safe["explanation"].startswith("SAFE:")
