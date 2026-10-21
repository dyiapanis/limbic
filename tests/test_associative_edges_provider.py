"""Associative links — provider wiring (boost re-rank, prefetch formation).

Stub-driven: real LimbicMemoryProvider methods (_search, prefetch) over fake
embedder/trust/storage doubles. Storage-contract details live in
test_associative_edges.py; these pin the SEAMS.
"""
import threading

from limbic.provider import LimbicMemoryProvider


class _FakeTrust:
    def __init__(self, conf_floor=0.1, rel_floor=0.1, trust_weight=0.0):
        self._vals = {
            "recall_confidence_floor": conf_floor,
            "relevance_floor": rel_floor,
            "trust_weight": trust_weight,
        }

    def get(self, key):
        return self._vals.get(key)

    def compute_activation(self, created_at, retrieval_count):
        return 1.0

    def is_correction(self, msg):
        return False

    def is_atrophy_candidate(self, *a, **k):
        return False

    def record_atrophy(self, *a, **k):
        pass

    def record_maturity(self, *a, **k):
        pass

    def tick(self, *a, **k):
        pass


class _FakeEmbedder:
    def embed(self, text, is_query=False):
        return [1.0]


class _RecordingStorage:
    """Minimal storage double: canned KNN results, records edge calls."""

    def __init__(self, facts, neighbours=None, fail_reinforce=False):
        self.facts = facts           # list[dict]: fact_id, content, vector_score
        self.neighbours = neighbours or {}
        self.fail_reinforce = fail_reinforce
        self.reinforced: list[tuple[list, str]] = []

    def search(self, embedding, filters, limit):
        return [dict(f) for f in self.facts]

    def search_fts(self, query_text, filters, limit):
        return []

    def search_graph(self, entity_names, filters, limit):
        return []

    def get_embedding(self, fact_id):
        return None

    def edge_neighbours(self, fact_id, pool_ids, user_id):
        return self.neighbours

    def open_supersessions(self, fact_ids):
        return {}  # contract grew in #19 — no canned supersessions here

    def reinforce_edges(self, fact_ids, user_id):
        self.reinforced.append((list(fact_ids), user_id))
        if self.fail_reinforce:
            raise RuntimeError("disk on fire")

    def update(self, fact_id, **fields):
        pass


def _mk_provider(config, storage):
    p = LimbicMemoryProvider()
    p._initialised = True
    p._config = config
    p._storage = storage
    p._embedder = _FakeEmbedder()
    p._trust = _FakeTrust()
    p._entity_extractor = None
    p._session_guardian_of = {}
    p._session_buffer = {}
    p._session_lock = threading.Lock()
    p._prefetch_latencies = []
    p._component_scores = {"embedding": [], "trust": [], "activation": []}
    p._get_user_id = lambda sid: "u1"
    p._evaluate_trust_signals = lambda *a, **k: None
    # Step-5 housekeeping (decay sweeps, atrophy, contradictions) is
    # orthogonal to the edge seam under test — no-op it out.
    p._apply_temporal_decay = lambda uid: None
    p._scan_and_sweep_gated_facts = lambda uid: None
    p._sweep_atrophy = lambda uid: None
    p._recheck_contradictions = lambda uid: None
    return p


def _fact(fid, score):
    return {
        "fact_id": fid,
        "content": f"fact {fid}",
        "vector_score": score,
        "created_at": "2026-10-20T00:00:00+00:00",
        "retrieval_count": 3,
        "trust_score": 1.0,
    }


def test_search_boost_reranks_within_slice():
    # budget 3: facts 1(0.7), 2(0.6), 3(0.5); fact 3 co-linked to fact 1 at
    # full weight → +0.2 lift puts it above fact 2, never above fact 1.
    storage = _RecordingStorage(
        [_fact(1, 0.7), _fact(2, 0.6), _fact(3, 0.5)],
        neighbours={3: 1.0},
    )
    p = _mk_provider({"associative_link_boost": 0.2, "context_budget": 3}, storage)
    out = p._search("q", "u1", limit=3)
    assert [f["fact_id"] for f in out] == [1, 3, 2]
    assert out[1]["link_boosted"] == 0.2  # observability field


def test_search_boost_off_by_default():
    storage = _RecordingStorage([_fact(1, 0.7), _fact(2, 0.6), _fact(3, 0.5)],
                                neighbours={3: 1.0})
    p = _mk_provider({}, storage)  # no key at all
    out = p._search("q", "u1", limit=3)
    assert [f["fact_id"] for f in out] == [1, 2, 3]
    assert storage.reinforced == []


def test_prefetch_forms_edges_on_real_retrieval():
    facts = [_fact(1, 0.7), _fact(2, 0.6), _fact(3, 0.5)]
    storage = _RecordingStorage(facts, neighbours={})
    p = _mk_provider({"associative_link_boost": 0.05, "context_budget": 3}, storage)
    p.prefetch("q", session_id="s")
    assert storage.reinforced and storage.reinforced[0][1] == "u1"
    assert set(storage.reinforced[0][0]) == {1, 2, 3}


def test_prefetch_abstain_forms_no_edges():
    # Top relevance below the confidence floor → no genuine surfacing → the
    # storage double must never see a reinforcement call. (The exact block
    # prefetch returns is not this seam's contract.)
    low = [_fact(1, 0.05), _fact(2, 0.04)]
    storage = _RecordingStorage(low)
    p = _mk_provider({"associative_link_boost": 0.05, "context_budget": 3}, storage)
    p._trust = _FakeTrust(conf_floor=0.3, rel_floor=0.1)
    p.prefetch("q", session_id="s")
    assert storage.reinforced == []


def test_prefetch_survives_reinforce_failure():
    storage = _RecordingStorage(
        [_fact(1, 0.7), _fact(2, 0.6)],
        fail_reinforce=True,
    )
    p = _mk_provider({"associative_link_boost": 0.05, "context_budget": 3}, storage)
    out = p.prefetch("q", session_id="s")  # must not raise
    assert isinstance(out, str)  # context block (possibly empty) — but no crash
    assert storage.reinforced  # the call was attempted