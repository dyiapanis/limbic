"""Supersession (#19) — recall prefers an open correction over the fact it
corrects, when both sides clear the relevance floor for the query.

Stub-driven: real LimbicMemoryProvider
._search over storage doubles. Storage-contract details live in
test_contradictions.py.
"""
import threading

from limbic.provider import LimbicMemoryProvider


class _FakeTrust:
    def __init__(self, conf_floor=0.1, rel_floor=0.1, trust_weight=0.2):
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
        # same first pattern as trust.CORRECTION_PATTERNS — good enough
        # to distinguish the cued/uncued test cases
        return "actually" in (msg or "").lower()

    def is_atrophy_candidate(self, *a, **k):
        return False

    def record_atrophy(self, *a, **k):
        pass

    def record_maturity(self, *a, **k):
        pass

    def tick(self, *a, **k):
        pass

    def surprise_percentile_value(self, *a, **k):
        return 1.1  # never triggers the surprise boost

    def correction_penalty_value(self, *a, **k):
        return 0.15

    def interference_penalty_value(self, *a, **k):
        return 0.20


class _FakeEmbedder:
    def embed(self, text, is_query=False):
        return [1.0]


class _FakeStorage:
    """Minimal storage double: canned KNN results, canned supersessions."""

    def __init__(self, facts, supers=None):
        self.facts = facts
        self.supers = supers or {}

    def search(self, embedding, filters, limit):
        return [dict(f) for f in self.facts]

    def search_fts(self, query_text, filters, limit):
        return []

    def get_embedding(self, fact_id):
        return None

    def open_supersessions(self, fact_ids):
        pool = set(fact_ids)
        return {o: n for o, n in self.supers.items() if o in pool}


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
    p._apply_temporal_decay = lambda uid: None
    p._scan_and_sweep_gated_facts = lambda uid: None
    p._sweep_atrophy = lambda uid: None
    p._recheck_contradictions = lambda uid: None
    return p


def _fact(fid, score, trust=1.0):
    return {
        "fact_id": fid,
        "content": f"fact {fid}",
        "vector_score": score,
        "created_at": "2026-10-20T00:00:00+00:00",
        "retrieval_count": 3,
        "trust_score": trust,
    }


def test_correction_wins_despite_higher_old_composite():
    # The production failure: old fact's composite (0.7, trust 1.0) beats
    # the correction (0.63, trust 0.5) — the 3% trust gap is smaller than
    # embedder noise. The open link must flip the outcome.
    facts = [_fact(1, 0.7, trust=1.0), _fact(2, 0.7, trust=0.5)]
    storage = _FakeStorage(facts, supers={1: 2})
    p = _mk_provider({"context_budget": 2}, storage)
    out = p._search("q", "u1", limit=2)
    assert [f["fact_id"] for f in out] == [2]


def test_safety_valve_old_surfaces_when_correction_below_floor():
    # Correction (id 2) is below the relevance floor → not in the gated
    # pool → the older fact must still surface: a noisy NLI label can
    # misorder at worst, never hide.
    facts = [_fact(1, 0.7, trust=1.0), _fact(2, 0.05, trust=0.5)]
    storage = _FakeStorage(facts, supers={1: 2})
    p = _mk_provider({"context_budget": 2}, storage)
    out = p._search("q", "u1", limit=2)
    assert [f["fact_id"] for f in out] == [1]


def test_no_link_original_ordering():
    facts = [_fact(1, 0.7, trust=1.0), _fact(2, 0.7, trust=0.5)]
    storage = _FakeStorage(facts, supers={})
    p = _mk_provider({"context_budget": 2}, storage)
    out = p._search("q", "u1", limit=2)
    assert [f["fact_id"] for f in out] == [1, 2]


def test_chain_keeps_latest():
    # a→b→c with all three gated: only the latest survives.
    facts = [_fact(1, 0.7), _fact(2, 0.7), _fact(3, 0.7)]
    storage = _FakeStorage(facts, supers={1: 2, 2: 3})
    p = _mk_provider({"context_budget": 3}, storage)
    out = p._search("q", "u1", limit=3)
    assert [f["fact_id"] for f in out] == [3]


# ── Write path: correction-cued NLI gate (#19 root cause) ────────


class _FakeNLI:
    def __init__(self, label="contradiction"):
        self.label = label
        self.calls = []

    def classify(self, existing, new):
        self.calls.append((existing, new))
        return self.label


class _WriteStorage:
    """Storage double for remember()'s interference scan."""

    def __init__(self, similar, next_id=2):
        self.similar = similar
        self.next_id = next_id
        self.stored = []
        self.updates = []
        self.links = []

    def search(self, embedding, filters, limit):
        return [dict(f) for f in self.similar]

    def get_all_embeddings(self, user_id):
        return []

    def store(self, content, user_id, embedding, metadata):
        fid = self.next_id
        self.next_id += 1
        self.stored.append((fid, content))
        return fid

    def link_contradiction(self, newer_fact_id, older_fact_id):
        self.links.append((newer_fact_id, older_fact_id))

    def update(self, fact_id, **fields):
        self.updates.append((fact_id, fields))

    def mark_atrophy(self, fact_id):
        pass


def _mk_write_provider(storage, nli):
    p = LimbicMemoryProvider()
    p._initialised = True
    p._config = {}
    p._storage = storage
    p._embedder = _FakeEmbedder()
    p._trust = _FakeTrust()
    p._nli = nli
    p._entity_extractor = None
    p._session_guardian_of = {}
    p._session_buffer = {}
    p._session_lock = threading.Lock()
    p._atrophy_marked_count = 0
    p._centroid_cache = {}
    p._centroid_write_count = {}
    p._get_user_id = lambda sid: "u1"
    return p


def _stale_fact(fid=1, score=0.65):
    return {
        "fact_id": fid,
        "content": "Her favourite font is JetBrains Mono",
        "score": score,
        "trust_score": 0.5,
        "last_retrieved": None,
        "created_at": "2026-10-20T00:00:00+00:00",
    }


def test_cued_correction_runs_nli_below_070():
    # The measured failure: value-swap corrections sit at 0.60-0.70 cosine
    # (0.6606 on the real pair) — below the normal gate, where NLI never
    # ran. A cued write must open the gate at 0.60.
    storage = _WriteStorage([_stale_fact(score=0.65)])
    nli = _FakeNLI()
    p = _mk_write_provider(storage, nli)
    res = p.remember("Actually, her favourite font is Iosevka", user_id="u1")
    assert res["status"] == "stored"
    assert len(nli.calls) == 1                      # NLI ran at 0.65
    assert storage.links == [(2, 1)]                # link formed
    assert storage.updates == [(1, {"trust_score": 0.35})]  # penalty applied


def test_uncued_write_keeps_070_gate():
    # Same similarity, no correction cue: the noise-management gate must
    # hold — NLI is noisy on unrelated pairs and stays out of the band.
    storage = _WriteStorage([_stale_fact(score=0.65)])
    nli = _FakeNLI()
    p = _mk_write_provider(storage, nli)
    res = p.remember("Her favourite font is Iosevka", user_id="u1")
    assert res["status"] == "stored"
    assert nli.calls == []                          # NLI not run
    assert storage.links == []                      # no link
    assert storage.updates == []                    # no penalty