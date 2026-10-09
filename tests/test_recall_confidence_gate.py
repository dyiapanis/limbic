
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
    """Provider with _search stubbed to return scored facts.

    Scores are dicts; a fact may carry separate relevance (cosine) and
    composite (trust-modulated) values — the gate reads RELEVANCE.
    """
    from limbic.provider import LimbicMemoryProvider
    p = LimbicMemoryProvider()
    p._initialised = True
    p._trust = _FakeTrust()
    p._session_guardian_of = {}
    p._get_user_id = lambda sid: "u1"

    def mk(i, s):
        if isinstance(s, dict):
            return {"fact_id": i, "content": f"fact {i}", **s}
        return {"fact_id": i, "content": f"fact {i}", "composite_score": s,
                "relevance": s}

    p._search = lambda query, uid, session_id="", user_message="": [
        mk(i, s) for i, s in enumerate(scores)
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


def test_natural_query_low_trust_high_cosine_passes():
    """Regression (review defect 2): gate reads cosine, not composite. A natural
    query hitting a fresh (low-trust) fact scores composite ~0.35 but relevance
    (cosine) ~0.72 — must PASS, not abstain."""
    p = _provider_with_results([{"composite_score": 0.35, "relevance": 0.72}])
    out = p.recall("tell me about the latest fix", session_id="s1")
    assert out["status"] == "ok"
    assert out["count"] == 1
    assert out["top_score"] == 0.72


def test_unanswerable_still_abstains_on_relevance():
    p = _provider_with_results([{"composite_score": 0.10, "relevance": 0.29}])
    out = p.recall("what is the weather in Tokyo", session_id="s1")
    assert out["status"] == "no_confident_match"


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


def test_high_relevance_new_fact_rescued_from_silent_pool():
    """The _search defect behind residual false-abstentions: a just-stored fact
    (activation < 0.5) with strong cosine must surface as active, not sit in the
    silent fallback pool while stale matured junk keeps `gated` non-empty."""
    from limbic.provider import LimbicMemoryProvider

    class _Trust(_FakeTrust):
        def compute_activation(self, created_at, retrieval_count):
            return 0.038  # maturation sigmoid early — new fact

    p = LimbicMemoryProvider()
    p._initialised = True
    p._trust = _Trust()
    p._session_guardian_of = {}
    p._get_user_id = lambda sid: "u1"
    p._config = {}

    pool = [
        # stale active junk: matured, mediocre cosine — passes floor
        {"fact_id": 1, "content": "old generic fact", "created_at": "2026-06-01",
         "retrieval_count": 9, "trust_score": 0.5, "vector_score": 0.30, "score": 0.30},
        # fresh, highly relevant: retrieval_count 0 -> silent under old logic
        {"fact_id": 2, "content": "the actual answer fact", "created_at": "2026-10-09",
         "retrieval_count": 0, "trust_score": 0.5, "vector_score": 0.534, "score": 0.534},
    ]
    p._storage = type("S", (), {
        "search": staticmethod(lambda embedding, filters, limit: pool),
        "search_fts": staticmethod(lambda query_text, filters, limit: []),
        "search_graph": staticmethod(lambda names, filters, limit: []),
        "get_embedding": staticmethod(lambda fid: None),
    })()
    import numpy as _np
    qv = _np.zeros(1024, dtype=_np.float32); qv[0] = 1.0
    v2 = _np.zeros(1024, dtype=_np.float32); v2[0] = 0.534  # cosine 0.534 vs query
    v1 = _np.zeros(1024, dtype=_np.float32); v1[0] = 0.30   # cosine 0.30
    p._embedder = type("E", (), {"embed": staticmethod(
        lambda q, is_query=False: qv.tolist())})
    from limbic.provider import LimbicMemoryProvider as _P
    p._cosine_distance = staticmethod(
        lambda a, b: 1.0 - float(_np.dot(a, b) / (_np.linalg.norm(a) * _np.linalg.norm(b) + 1e-12)))
    # storage returns vec rows with vector_score; get_embedding returns stored vecs
    p._storage.get_embedding = staticmethod(
        lambda fid: v2.tolist() if fid == 2 else (v1.tolist() if fid == 1 else None))

    # REAL _search — the rescue lives inside its scoring loop:
    facts = LimbicMemoryProvider._search(p, "q", "u1")
    by_id = {f["fact_id"]: f for f in facts}
    assert "relevance" in by_id[2], "high-relevance new fact not surfaced (still silent-pooled)"
    assert by_id[2]["relevance"] == 0.534
