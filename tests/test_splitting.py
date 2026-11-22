"""Atomic soft-split (sprint C5) — detector, sibling-exempt write, recall link.

Model-free: real SQLiteStorage (dims=8 / WORD_DIMS), fake embedder, fake NLI.

Gates covered here:
- structure detector cases: numbered / md headers / paragraph clusters /
  false-positive single narrative / small-fragment merge
- REAL multi-section audit fixture: ALL children stored,
  ZERO NLI penalties on pre-existing facts
- sibling-exempt write path: near-identical siblings both land untouched
  (the plain sequential path would dock the first or drop the second)
- recall semantics: children retrievable independently, parent link carried
  in recall results, parent row never surfaces
"""
import math
import os
import re
import sqlite3
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.provider import LimbicMemoryProvider  # noqa: E402
from limbic.splitting import split_structured  # noqa: E402
from limbic.store_sqlite import SQLiteStorage  # noqa: E402

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "audit_fact25.txt")


# ── fake embedder: signed word-hash — same words cluster, unrelated text ──
# is near-orthogonal. Deterministic (md5), never Python's random hash().

WORD_DIMS = 64


class WordEmbedder:
    def __init__(self, dims=WORD_DIMS):
        self.dims = dims
        self.calls = 0

    def _vec(self, text):
        v = [0.0] * self.dims
        for w in re.findall(r"[a-z0-9_]+", (text or "").lower()):
            h = int(__import__("hashlib").md5(w.encode()).hexdigest(), 16)
            v[h % self.dims] += 1.0 if h & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, content, is_query=False):
        self.calls += 1
        return self._vec(content)

    @property
    def cache_hits(self):
        return 0


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


class FakeTrust:
    """Floors deliberately low so recall surfaces whatever matched."""

    def __init__(self, conf_floor=0.05, rel_floor=0.05):
        self._vals = {
            "recall_confidence_floor": conf_floor,
            "relevance_floor": rel_floor,
            "trust_weight": 0.0,
        }

    def get(self, key):
        return self._vals.get(key)

    def compute_activation(self, created_at, retrieval_count):
        return 1.0

    def is_correction(self, msg):
        return "actually," in (msg or "").lower()

    def is_atrophy_candidate(self, *a, **k):
        return False

    def record_atrophy(self, *a, **k):
        pass

    def record_maturity(self, *a, **k):
        pass

    def tick(self, *a, **k):
        pass

    def surprise_percentile_value(self, *a, **k):
        return 1.1  # never boosts

    def correction_penalty_value(self, *a, **k):
        return 0.15

    def interference_penalty_value(self, *a, **k):
        return 0.20

    def interference_floor_value(self, *a, **k):
        return 0.0


class FakeNLI:
    def classify(self, old, new):
        return "neutral", 0.0


def _mk_provider(tmp_path, embedder=None):
    """House pattern (test_provider_tools/test_supersession): real provider,
    real SQLiteStorage, fake models."""
    p = LimbicMemoryProvider()
    p._load_config(str(tmp_path))
    p._embedder = embedder or WordEmbedder()
    p._nli = FakeNLI()
    p._entity_extractor = None
    p._trust = FakeTrust()
    p._storage = SQLiteStorage(str(tmp_path / "limbic.db"), dims=WORD_DIMS)
    p._initialised = True
    p._session_guardian_of = {}
    p._session_buffer = {}
    p._session_lock = threading.Lock()
    p._prefetch_latencies = []
    p._component_scores = {"embedding": [], "trust": [], "activation": []}
    p._get_user_id = lambda sid: "grace"
    p._evaluate_trust_signals = lambda *a, **k: None
    p._apply_temporal_decay = lambda uid: None
    p._scan_and_sweep_gated_facts = lambda uid: None
    p._sweep_atrophy = lambda uid: None
    p._recheck_contradictions = lambda uid: None
    return p


@pytest.fixture()
def provider(tmp_path):
    p = _mk_provider(tmp_path)
    yield p
    p._storage.close()


@pytest.fixture()
def store(tmp_path):
    s = SQLiteStorage(str(tmp_path / "limbic.db"), dims=WORD_DIMS)
    yield s
    s.close()


def _vec_rows_in_index(s: SQLiteStorage) -> list[int]:
    db = s._db
    return [r[0] for r in db.execute("SELECT rowid FROM vec_index").fetchall()]


def _role(s, fid):
    return s._db.execute(
        "SELECT split_role, split_parent_id FROM facts WHERE fact_id = ?",
        (fid,)).fetchone()


# ────────────────────────── detector cases ──────────────────────────

def test_real_audit_fact_25_splits_into_sections():
    content = open(FIXTURE).read()
    assert len(content) > 700
    frags = split_structured(content)
    assert frags is not None and len(frags) == 6   # preamble + items 1..5
    assert all(len(f) >= 120 for f in frags)
    # every numbered stage keeps its own fragment
    for stage in ("1. SOURCE", "2. REFERENCE SET", "3. PANELS",
                  "4. FIXES", "5. MOTION"):
        assert any(f.startswith(stage) for f in frags)


def test_detector_numbered_and_headers_and_clusters():
    numbered = "\n".join(
        f"{i}. {label} " + "detail " * 40
        for i, label in enumerate(["alpha", "bravo", "charlie"], 1))
    assert len(numbered) > 700 and len(split_structured(numbered)) == 3

    headers = "\n\n".join(
        f"## {label}\n" + "prose " * 80
        for label in ("design", "rollout"))
    assert len(split_structured(headers)) == 2

    paras = "\n\n".join("paragraph of prose " * 30 for _ in range(3))
    frags = split_structured(paras)
    assert frags is not None and len(frags) == 3


def test_single_narrative_false_positive_stays_whole():
    single = ("One continuous narrative about a single event unfolding with "
              "no sections and no lists and no paragraph breaks at all, "
              "just flowing prose. " * 10).strip()
    assert len(single) > 700
    assert split_structured(single) is None
    # below the length gate: never split, however marked
    assert split_structured("- one\n- two") is None


def test_tiny_fragments_merge_to_min_child():
    # 3 items, two of them below min_child — they must fold into neighbours
    text = ("Long preamble that is itself long enough to stand alone. " * 5
            + "\n1. tiny\n2. small\n3. "
            + "this item is long enough alone " * 14)
    frags = split_structured(text)
    assert frags is not None and len(frags) == 2
    assert all(len(f) >= 120 for f in frags)


# ─────────────────────── store-level split write ───────────────────────

def _children_store_payload(s, contents, contradictions=None):
    return [
        {"content": c, "embedding": s2v(c, s), "trust_score": 0.5,
         "contradictions": (contradictions or {}).get(c, [])}
        for c in contents
    ]


def s2v(text, s):
    v = [0.0] * WORD_DIMS
    for w in re.findall(r"[a-z0-9_]+", text.lower()):
        v[(len(w) * 7) % WORD_DIMS] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def test_store_split_parent_provenance_excluded_from_lanes(store):
    parent_text = ("PARENT preamble unique words qwertyuiop\n"
                   "1. child one alpha content " + "alpha detail " * 10 +
                   "\n2. child two beta content " + "beta detail " * 10)
    children = _children_store_payload(
        store,
        ["1. child one alpha content " + "alpha detail " * 10,
         "2. child two beta content " + "beta detail " * 10])
    out = store.store_split(parent_text, "grace", children)
    parent, child_ids = out["parent_fact_id"], out["child_fact_ids"]
    assert len(child_ids) == 2

    row = _role(store, parent)
    assert row["split_role"] == "parent" and row["split_parent_id"] is None
    assert store.get(parent)["content"] == parent_text
    for cid, ch in zip(child_ids, children):
        r = _role(store, cid)
        assert r["split_role"] == "child" and r["split_parent_id"] == parent
        assert store.get(cid)["content"] == ch["content"]

    idx = set(_vec_rows_in_index(store))
    assert parent not in idx and set(child_ids) <= idx

    # FTS lane: parent text matches, but parent row must not surface
    hits = store.search_fts(query_text="qwertyuiop", filters={"user_id": "grace"},
                            limit=5)
    assert parent not in [h.get("fact_id") for h in hits]
    # graph lane: entity pinned to BOTH parent and child — the split-role
    # guard must surface the child only
    store.store_entities(parent, [{"name": "Shared Entity", "type": "test"}])
    store.store_entities(child_ids[0], [{"name": "Shared Entity", "type": "test"}])
    got = store.search_graph(["Shared Entity"], {"user_id": "grace"}, limit=5)
    got_ids = [g.get("fact_id") for g in got]
    assert parent not in got_ids and child_ids[0] in got_ids
    # vector recall still returns the children independently
    vec_hits = store.search(embedding=children[0]["embedding"],
                            filters={"user_id": "grace"}, limit=5)
    assert child_ids[0] == vec_hits[0]["fact_id"]
    assert vec_hits[0].get("split_parent_id") == parent


def test_store_split_is_atomic_on_mid_batch_failure(store):
    before = len(store.export_user("grace")["facts"])
    good = "1. child one alpha content " + "alpha detail " * 10
    children = [
        {"content": good, "embedding": s2v(good, store), "trust_score": 0.5,
         "contradictions": []},
        {"content": "2. broken", "embedding": [1.0] * 3,   # wrong dims
         "trust_score": 0.5, "contradictions": []},
    ]
    with pytest.raises(sqlite3.Error):
        store.store_split("PARENT full text " * 30 + good, "grace", children)
    after = store.export_user("grace")["facts"]
    assert len(after) == before          # parent and sibling rolled back
    assert _vec_rows_in_index(store) == []


# ─────────────────── provider-level integration (gates) ───────────────────

def test_real_audit_fixture_all_children_stored_zero_penalties(provider):
    s = provider._storage
    # one PRE-EXISTING unrelated fact — must end with trust untouched
    unrelated = "The office printer lives on the second floor near the stairs."
    pre_id = s.store(content=unrelated, user_id="grace",
                     embedding=WordEmbedder().embed(unrelated),
                     metadata={"trust_score": 0.5})
    pre_trust = s.get(pre_id)["trust_score"]

    content = open(FIXTURE).read()
    frags = split_structured(content)
    out = provider.remember(content, user_id="grace")

    assert out["status"] == "stored_split"
    assert len(out["child_fact_ids"]) == len(frags)          # ALL children
    parent = out["parent_fact_id"]

    assert _role(s, parent)["split_role"] == "parent"
    assert s.get(parent)["content"] == content
    for cid in out["child_fact_ids"]:
        assert _role(s, cid)["split_parent_id"] == parent
    # zero NLI penalties: pre-existing fact's trust is untouched
    assert s.get(pre_id)["trust_score"] == pre_trust == 0.5
    # parent carries no vector → can never surface in KNN recall
    assert parent not in set(_vec_rows_in_index(s))
    assert set(out["child_fact_ids"]) <= set(_vec_rows_in_index(s))


def test_sibling_exemption_near_identical_sections_both_land(tmp_path):
    pad = " the deployment team rehearses the failover drill each morning"
    sec = ("1. Drill:" + pad * 6 + " without exception reminders.")
    near = ("2. Drill:" + pad * 6 + " without exception notes.")
    e = WordEmbedder()
    assert cosine(e.embed(sec), e.embed(near)) > 0.85   # the hostile band

    p = _mk_provider(tmp_path)
    out = p.remember("Squad drill log.\n" + sec + "\n" + near, user_id="grace")
    assert out["status"] == "stored_split"
    a, b = out["child_fact_ids"]
    s = p._storage
    assert cosine(e.embed(s.get(a)["content"]),
                  e.embed(s.get(b)["content"])) > 0.85
    # BOTH siblings untouched at 0.5 — no interference dock, no drop
    assert s.get(a)["trust_score"] == 0.5
    assert s.get(b)["trust_score"] == 0.5
    assert b in _vec_rows_in_index(p._storage)          # second sibling stored

    # counterfactual: the PLAIN sequential path on the same pair docks the
    # first sibling (neutral >0.85 → interference) — the split path did not.
    p2 = _mk_provider(tmp_path / "plain")
    p2.remember(sec, user_id="grace")
    p2.remember(near, user_id="grace")
    assert p2._storage.export_user("grace")["fact_count"] == 2
    rows = p2._storage.export_user("grace")["facts"]
    first = min(r["fact_id"] for r in rows)
    assert p2._storage.get(first)["trust_score"] < 0.5  # docked by the plain path
    p._storage.close()
    p2._storage.close()


def test_recall_children_retrievable_parent_link_carried_parent_hidden(provider):
    content = open(FIXTURE).read()
    out = provider.remember(content, user_id="grace")
    parent = out["parent_fact_id"]

    res = provider.recall("PANELS ideogram character reference_image_urls",
                          session_id="sess")
    assert res["status"] == "ok" and res["results"]
    top = res["results"][0]
    assert top.get("parent_fact_id") == parent     # parent link carried
    assert all(len(r["content"]) < len(content)    # parent never surfaces
               for r in res["results"])
