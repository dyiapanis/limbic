"""#20 items 2 & 4 — abstention diagnostics + calibration watch.

Model-free (fake embedder/NLI, real SQLiteStorage) — same fixture pattern
as test_forget_tool.py.

Covers:
  - floor-case abstention carries floor, path, best_excerpt (no ids)
  - empty-case abstention carries floor, path="none", no excerpt
  - near-miss content never enters the ok-path results
  - recall outcome counters bump; stats() exposes abstention_rate
  - metrics fail soft (a backend without metric support changes nothing)
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.provider import LimbicMemoryProvider  # noqa: E402
from limbic.store_sqlite import SQLiteStorage  # noqa: E402
from limbic.trust import TrustEngine  # noqa: E402

DIMS = 8


class FakeEmbedder:
    def __init__(self, dims=DIMS):
        self.dims = dims

    def embed(self, content, is_query=False):
        v = [float((ord(c) % 13) + 1) for c in (content or "x")[:DIMS]]
        v += [1.0] * (self.dims - len(v))
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    @property
    def cache_hits(self) -> int:
        return 0


class FakeNLI:
    def classify(self, old, new):
        return "neutral", 0.0


@pytest.fixture()
def provider(tmp_path):
    p = LimbicMemoryProvider()
    p._load_config(str(tmp_path))
    p._embedder = FakeEmbedder()
    p._nli = FakeNLI()
    p._entity_extractor = None
    p._trust = TrustEngine(str(tmp_path / "limbic_params.json"))
    p._storage = SQLiteStorage(str(tmp_path / "limbic.db"), dims=DIMS)
    p._initialised = True
    yield p
    p._storage.close()


def test_floor_abstention_carries_diagnostics(provider):
    provider.remember("the zanzibar ferry leaves at six", user_id="u1")
    out = provider.recall("xyzzyplugh", session_id="s")
    # whatever the gate decides, the payload contract holds:
    assert "floor" in out and "path" in out and "best_excerpt" in out
    assert out["floor"] == pytest.approx(0.50)
    if out["status"] == "no_confident_match":
        assert out["top_score"] >= 0.0
        # near-miss case: excerpt + lane present, never an id
        assert "fact_id" not in out
        if out["best_excerpt"] is not None:
            assert isinstance(out["best_excerpt"], str)


def test_empty_results_abstention(provider, tmp_path):
    # A user with zero facts: _search returns nothing → abstention with path none
    provider._get_user_id = lambda sid: "ghost"
    out = provider.recall("anything at all", session_id="s")
    assert out["status"] == "no_confident_match"
    assert out["path"] == "none"
    assert out["best_excerpt"] is None
    assert out["floor"] == pytest.approx(0.50)


def test_counters_bump_and_stats_exposes_rate(provider):
    provider._get_user_id = lambda sid: "u1"
    provider.remember("counted fact about the red bicycle", user_id="u1")
    r_ok = provider.recall("red bicycle", session_id="s")
    r_miss = provider.recall("completely unrelated garble", session_id="s")
    total = provider._storage.get_metrics()["recall_total"]
    assert total >= 2
    if r_miss["status"] == "no_confident_match" and r_ok["status"] == "ok":
        assert provider._storage.get_metrics()["recall_abstain"] >= 1
    st = provider.stats(user_id="u1")
    assert st["recall"]["total"] == total
    assert isinstance(st["recall"]["abstention_rate"], float) or st["recall"]["abstention_rate"] is None
    agg = provider.stats()
    assert agg["recall"]["total"] == total


def test_metrics_fail_soft(provider, monkeypatch):
    provider._get_user_id = lambda sid: "u1"
    # a storage backend that rejects metrics entirely must not break recall
    monkeypatch.setattr(provider._storage, "bump_metric",
                        lambda k, n=1: (_ for _ in ()).throw(RuntimeError("no metrics")))
    monkeypatch.setattr(provider._storage, "get_metrics",
                        lambda: (_ for _ in ()).throw(RuntimeError("no metrics")))
    provider.remember("fact with broken metrics", user_id="u1")
    out = provider.recall("broken metrics", session_id="s")
    assert out["status"] in ("ok", "no_confident_match")  # recall still works
    st = provider.stats(user_id="u1")
    assert st["recall"]["total"] == 0  # degraded to zeros, no crash