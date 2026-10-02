"""Provider-level continuing-correction checks.

`test_contradictions.py` covers the junction table. These cover the parts that
need a provider instance: the write path that records links, and the tick pass
that re-asserts or resolves them.

Model-free — embedder and NLI are stubbed, so this runs on a bare checkout
without downloading the ONNX models.
"""
import logging
import threading

import pytest

from limbic.provider import LimbicMemoryProvider
from limbic.store_sqlite import SQLiteStorage
from limbic.trust import TrustEngine

DIMS = 8
logging.disable(logging.CRITICAL)


class StubEmbedder:
    """Deterministic: same text -> same vector, so similarity is predictable."""

    def embed(self, text):
        return [abs(hash(text)) % 100 / 100.0] * DIMS


class StubNLI:
    def __init__(self, label="contradiction"):
        self.label = label
        self.calls = 0

    def classify(self, a, b):
        self.calls += 1
        return self.label


def make_provider(tmp_path, nli_label="contradiction", max_per_turn=3):
    """A provider with real storage/trust and stubbed model layers."""
    store = SQLiteStorage(str(tmp_path / "limbic.db"), dims=DIMS)
    trust = TrustEngine(str(tmp_path / "limbic_params.json"))
    trust._set("max_corrections_per_turn", max_per_turn)
    trust._set("relevance_floor", 0.25)

    p = LimbicMemoryProvider.__new__(LimbicMemoryProvider)
    p._storage = store
    p._trust = trust
    p._nli = StubNLI(nli_label)
    p._embedder = StubEmbedder()
    p._entity_extractor = None
    p._initialised = True
    p._lock = threading.RLock()
    p._config = {}
    p._session_buffer = {}
    p._session_lock = threading.RLock()
    p._component_scores = {"embedding": [], "trust": [], "activation": []}
    p._centroid_cache = {}
    p._centroid_writes = 0
    p._centroid_cache_hits = 0
    p._centroid_cache_misses = 0
    p._last_recall_count = 0
    p._atrophy_marked_count = 0
    p._atrophy_swept_count = 0
    # isolate the unit under test from unrelated collaborators
    p._get_user_id = lambda sid: "alice"
    p._detect_subject_user = lambda content, uid: None
    p._is_ephemeral_date_fact = lambda c: False
    p._compute_surprise = lambda emb, uid: 0.5
    p._check_organic_atrophy = lambda *a, **k: None
    return p, store, trust


# ── Write path ────────────────────────────────────────────────────────

def test_contradiction_creates_a_link(tmp_path):
    """NLI fires before the new fact is stored, so the link can only be written
    after store() returns an id. If that ordering regresses, this fails."""
    p, store, _ = make_provider(tmp_path)
    first = p.remember("Alice prefers Postgres for the analytics service", user_id="alice")
    second = p.remember("Alice no longer prefers Postgres for analytics", user_id="alice")

    assert first["status"] == "stored"
    assert second["status"] == "stored"

    links = store.unresolved_contradictions("alice", limit=10)
    assert len(links) == 1
    assert links[0]["newer_fact_id"] == second["fact_id"]   # the correction
    assert links[0]["older_fact_id"] == first["fact_id"]    # the corrected


def test_write_time_penalty_still_applies(tmp_path):
    """The link is additive — the existing one-shot penalty must be unchanged."""
    p, store, _ = make_provider(tmp_path)
    first = p.remember("Alice prefers Postgres for the analytics service", user_id="alice")
    p.remember("Alice no longer prefers Postgres for analytics", user_id="alice")
    assert store.get(first["fact_id"])["trust_score"] < 0.5


def test_no_link_without_contradiction(tmp_path):
    p, store, _ = make_provider(tmp_path, nli_label="entailment")
    p.remember("Alice prefers Postgres for the analytics service", user_id="alice")
    p.remember("Alice likes Postgres for the analytics service", user_id="alice")
    assert store.count_open_contradictions("alice") == 0


# ── Tick re-check ─────────────────────────────────────────────────────

def test_recheck_resolves_when_correction_landed(tmp_path):
    """Older fact faded below the floor -> the correction worked, close the link."""
    p, store, _ = make_provider(tmp_path)
    newer, older = _link(p, store, older_trust=0.10)
    p._recheck_contradictions("alice")
    assert store.count_open_contradictions("alice") == 0
    assert store.get(older)["trust_score"] == pytest.approx(0.10)   # untouched


def test_recheck_resolves_when_no_longer_contradictory(tmp_path):
    p, store, _ = make_provider(tmp_path, nli_label="neutral")
    newer, older = _link(p, store, older_trust=0.9)
    p._recheck_contradictions("alice")
    assert store.count_open_contradictions("alice") == 0
    assert store.get(older)["trust_score"] == pytest.approx(0.9)    # untouched


def test_recheck_reasserts_when_trust_recovered(tmp_path):
    """The core case: continuations outvoted the penalty, NLI still contradicts."""
    p, store, trust = make_provider(tmp_path)
    newer, older = _link(p, store, older_trust=0.9)
    penalty = trust.correction_penalty_value()

    p._recheck_contradictions("alice")

    assert store.get(older)["trust_score"] == pytest.approx(0.9 - penalty)
    # stays open so the correction keeps acting
    assert store.count_open_contradictions("alice") == 1


def test_recheck_is_bounded(tmp_path):
    """Writes and NLI calls per turn are capped — the whole point of the design."""
    p, store, _ = make_provider(tmp_path, max_per_turn=2)
    newer = _fact(store, 99)
    for i in range(6):
        store.link_contradiction(newer, _fact(store, i, trust=0.9))

    p._recheck_contradictions("alice")

    assert p._nli.calls <= 2
    assert store.count_open_contradictions("alice") == 6   # none resolved


def test_recheck_expires_stale_links(tmp_path):
    p, store, _ = make_provider(tmp_path)
    newer, older = _link(p, store, older_trust=0.9)
    store._db.execute(
        "UPDATE contradictions SET created_at='2020-01-01T00:00:00+00:00' WHERE resolved_at IS NULL"
    )
    store._db.commit()

    p._recheck_contradictions("alice")

    assert store._db.execute("SELECT COUNT(*) FROM contradictions").fetchone()[0] == 0


def test_recheck_is_safe_to_repeat(tmp_path):
    p, store, _ = make_provider(tmp_path)
    newer, older = _link(p, store, older_trust=0.9)
    p._recheck_contradictions("alice")
    first = store.get(older)["trust_score"]
    p._recheck_contradictions("alice")
    assert store.get(older)["trust_score"] <= first
    assert store.count_open_contradictions("alice") == 1


def test_recheck_no_links_is_a_noop(tmp_path):
    p, store, _ = make_provider(tmp_path)
    _fact(store, 1)
    p._recheck_contradictions("alice")          # must not raise
    assert store.count_open_contradictions("alice") == 0


# ── helpers ───────────────────────────────────────────────────────────

def _fact(store, i, user="alice", trust=0.5):
    return store.store(
        content=f"fact number {i}",
        user_id=user,
        embedding=[float(i) / 10] * DIMS,
        metadata={"trust_score": trust, "created_at": "2026-09-01T00:00:00+00:00"},
    )


def _link(p, store, older_trust=0.5):
    """Create a newer/older pair with an open contradiction link between them."""
    older = _fact(store, 1, trust=older_trust)
    newer = _fact(store, 2)
    store.link_contradiction(newer, older)
    return newer, older
