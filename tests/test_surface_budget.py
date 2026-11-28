"""Surface budget — rank-ordered hard cap on surfaced tokens (SPEC: 800)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from limbic.provider import (  # noqa: E402
    SURFACE_BUDGET_TOKENS,
    _apply_surface_budget,
    _estimate_tokens,
)


class _FakeTrust:
    def get(self, name):
        if name == "recall_confidence_floor":
            return 0.40
        if name == "relevance_floor":
            return 0.25
        return 0.0

    def compute_activation(self, created_at, retrieval_count):
        return 0.9  # active


def _fact(i, chars, score=0.8):
    return {"fact_id": i, "content": "x" * chars,
            "composite_score": score, "relevance": score}


def _provider_with_results(facts):
    from limbic.provider import LimbicMemoryProvider
    p = LimbicMemoryProvider()
    p._initialised = True
    p._trust = _FakeTrust()
    p._session_guardian_of = {}
    p._get_user_id = lambda sid: "u1"
    p._search = lambda query, uid, session_id="", user_message="": list(facts)
    return p


# ── estimator ─────────────────────────────────────────────────────────

def test_estimate_tokens_matches_benchmark_estimator():
    # chars//4 — the same instrument the retrieval benchmark uses.
    assert _estimate_tokens("x" * 400) == 100
    assert _estimate_tokens("x" * 7) == 1
    assert _estimate_tokens("") == 0


# ── selection ─────────────────────────────────────────────────────────

def test_under_budget_passthrough_is_identity():
    facts = [_fact(i, 400) for i in range(5)]  # 5 × 100 tok = 500 < 800
    out = _apply_surface_budget(facts)
    assert out == facts
    assert out[0] is facts[0]  # same objects, no copies, order kept


def test_over_budget_caps_total_tokens():
    facts = [_fact(i, 1200) for i in range(10)]  # 10 × 300 tok = 3000
    out = _apply_surface_budget(facts)
    total = sum(_estimate_tokens(f["content"]) for f in out)
    assert total <= SURFACE_BUDGET_TOKENS
    assert total > 0


def test_rank_order_preserved_head_facts_whole():
    facts = [_fact(i, 600) for i in range(10)]  # 150 tok each
    out = _apply_surface_budget(facts)
    # 5 whole facts = 750 tok; 50 remaining → 6th fact trimmed to fit,
    # 7th+ stay dark.
    assert [f["fact_id"] for f in out] == [0, 1, 2, 3, 4, 5]
    for f in out[:-1]:
        assert f["content"] == "x" * 600  # whole, untrimmed
    assert out[-1]["content"].endswith("…")


def test_tail_trimmed_to_fit_with_ellipsis():
    facts = [_fact(0, 600), _fact(1, 10000)]  # 150 tok, then 2500 tok
    out = _apply_surface_budget(facts)
    assert len(out) == 2
    tail = out[1]["content"]
    assert tail.endswith("…")
    assert len(tail) < 10000
    total = sum(_estimate_tokens(f["content"]) for f in out)
    assert total <= SURFACE_BUDGET_TOKENS


def test_subsliver_tail_left_dark():
    # 200 tok × 4 = 800 exactly; next fact gets 0 remaining → dropped whole.
    facts = [_fact(i, 800) for i in range(4)] + [_fact(4, 40)]
    out = _apply_surface_budget(facts)
    assert [f["fact_id"] for f in out] == [0, 1, 2, 3]
    assert all(f["content"] == "x" * 800 for f in out)  # no sliver, no trim


def test_input_never_mutated():
    facts = [_fact(i, 1200) for i in range(5)]
    snapshot = [(f["fact_id"], f["content"]) for f in facts]
    _apply_surface_budget(facts)
    assert [(f["fact_id"], f["content"]) for f in facts] == snapshot


def test_single_oversized_fact_surfaces_trimmed():
    facts = [_fact(0, 8000)]  # 2000 tok alone
    out = _apply_surface_budget(facts)
    assert len(out) == 1
    assert out[0]["content"].endswith("…")
    assert _estimate_tokens(out[0]["content"]) <= SURFACE_BUDGET_TOKENS


def test_empty_and_none_content():
    assert _apply_surface_budget([]) == []
    out = _apply_surface_budget([{"fact_id": 1, "content": None,
                                  "composite_score": 0.8}])
    assert len(out) == 1  # empty content costs nothing, still surfaces


# ── wired into recall() ───────────────────────────────────────────────

def test_recall_payload_capped():
    facts = [_fact(i, 1200) for i in range(10)]  # 3000 tok total
    out = _provider_with_results(facts).recall("real query", session_id="s1")
    assert out["status"] == "ok"
    payload = sum(len(r["content"]) // 4 for r in out["results"])
    assert payload <= SURFACE_BUDGET_TOKENS
    assert out["count"] == len(out["results"])


def test_recall_under_budget_unchanged():
    facts = [_fact(i, 400) for i in range(3)]  # 300 tok total
    out = _provider_with_results(facts).recall("real query", session_id="s1")
    assert out["status"] == "ok"
    assert out["count"] == 3
    assert [r["content"] for r in out["results"]] == ["x" * 400] * 3


def test_recall_abstain_path_has_no_budget_interaction():
    # Abstention runs on the uncut pool; budget never creates an abstain.
    facts = [_fact(0, 8000, score=0.31)]  # below 0.40 floor
    out = _provider_with_results(facts).recall("real query", session_id="s1")
    assert out["status"] == "no_confident_match"
    assert out["results"] == []
    assert out["top_score"] == 0.31
