
"""Recall-confidence gate (no_confident_match abstention) — read-side metacognition."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class _FakeTrust:
    def __init__(self, floor=0.40):
        self._floor = floor

    def get(self, name):
        if name == "recall_confidence_floor":
            return self._floor
        if name == "relevance_floor":
            return 0.25
        return 0.0


def _provider_with_results(scores):
    """Provider with _search stubbed to return scored facts."""
    from limbic.provider import LimbicMemoryProvider
    p = LimbicMemoryProvider()
    p._initialised = True
    p._trust = _FakeTrust()
    p._session_guardian_of = {}
    p._get_user_id = lambda sid: "u1"
    p._search = lambda query, uid, session_id="", user_message="": [
        {"fact_id": i, "content": f"fact {i}", "composite_score": s}
        for i, s in enumerate(scores)
    ]
    return p


def test_recall_abstains_below_floor():
    p = _provider_with_results([0.31, 0.28, 0.12])  # all below 0.40
    out = p.recall("any query", session_id="s1")
    assert out["status"] == "no_confident_match"
    assert out["results"] == []
    assert out["count"] == 0
    assert out["top_score"] == 0.31


def test_recall_passes_above_floor():
    p = _provider_with_results([0.83, 0.55])
    out = p.recall("real query", session_id="s1")
    assert out["status"] == "ok"
    assert out["count"] == 2
    assert out["top_score"] == 0.83


def test_recall_empty_search_abstains():
    p = _provider_with_results([])
    out = p.recall("anything", session_id="s1")
    assert out["status"] == "no_confident_match"
    assert out["top_score"] == 0.0


def test_gate_param_registered():
    from limbic.trust import DEFAULT_PARAMS
    entry = DEFAULT_PARAMS["recall_confidence_floor"]
    assert entry["value"] == 0.40 and entry["min"] < entry["value"] < entry["max"]


def test_existing_params_file_backfills_default():
    """A params JSON written before the gate must not crash TrustEngine init."""
    import json, tempfile
    from limbic.trust import TrustEngine
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "limbic_params.json")
        json.dump({"relevance_floor": {"value": 0.25, "min": 0.15, "max": 0.60}}, open(path, "w"))
        t = TrustEngine(path)
        assert t.get("recall_confidence_floor") == 0.40  # backfilled
